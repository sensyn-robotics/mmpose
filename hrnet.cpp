#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "platform.h"
#include "net.h"
#include "omp.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

struct Keypoints {
    float x;
    float y;

    float score;
    Keypoints() : x(0), y(0), score(0) {}

    Keypoints(float x, float y, float score) : x(x), y(y), score(score) {}
};

struct Box {
    float center_x;
    float center_y;
    float scale_x;
    float scale_y;
    float scale_prob;
    float score;

    Box() : center_x(0), center_y(0), scale_x(0), scale_y(0), scale_prob(0), score(0) {}

    Box(float center_x, float center_y, float scale_x, float scale_y, float scale_prob, float score):
        center_x(center_x), center_y(center_y), scale_x(scale_x), scale_y(scale_y), scale_prob(scale_prob), score(score) {}
   
};

void bbox_xywh2cs(float bbox[], float aspect_ratio, float padding, float pixel_std, float *center, float *scale) {
    float x = bbox[0];
    float y = bbox[1];
    float w = bbox[2];
    float h = bbox[3];
    *center = x + w * 0.5;
    *(center + 1) = y + h * 0.5;
    if (w > aspect_ratio * h)
        h = w * 1.0 / aspect_ratio;
    else if (w < aspect_ratio * h)
        w = h * aspect_ratio;
 
 
    *scale = (w / pixel_std) * padding;
    *(scale + 1) = (h / pixel_std) * padding;
}


void rotate_point(float *point, float angle_rad, float *rotated_point) {
    float x_rotated = point[0] * cos(angle_rad) - point[1] * sin(angle_rad);
    float y_rotated = point[0] * sin(angle_rad) + point[1] * cos(angle_rad);

    rotated_point[0] = std::move(x_rotated);
    rotated_point[1] = std::move(y_rotated);

}

void _get_3rd_point(cv::Point2f a, cv::Point2f b, float *direction){
    direction[0] = b.x - (a.y - b.y);
    direction[1] = b.y + (a.x - b.x);
}

void get_affine_transform(float *center, float *scale, float rot, float *output_size, float *shift, bool inv, cv::Mat &trans) {
    float scale_tmp[] = {0, 0};
    scale_tmp[0] = scale[0] * 200.0;
    scale_tmp[1] = scale[1] * 200.0;

    float src_w = scale_tmp[0];
    float dst_w = output_size[0];
    float dst_h = output_size[1];
    float rot_rad = M_PI * rot / 180;
    float pt[] = {0, 0};
    pt[0] = 0;
    pt[1] = src_w * (-0.5);
    float src_dir[] = {0,0};
    rotate_point(pt, rot_rad, src_dir);
    
    float dst_dir[] = {0,0};
    dst_dir[0] = 0;
    dst_dir[1] = dst_w * (-0.5);

    cv::Point2f src[3] = {cv::Point2f(0,0), cv::Point2f(0,0), cv::Point2f(0,0)};
    src[0] = cv::Point2f( center[0] + scale_tmp[0] * shift[0], center[1] + scale_tmp[1] * shift[1]);
    src[1] = cv::Point2f(center[0] + src_dir[0] + scale_tmp[0] * shift[0],
                center[1] + src_dir[1] + scale_tmp[1] * shift[1]);

    float direction_src[] = {0, 0};
    _get_3rd_point(src[0], src[1], direction_src);
    src[2] = cv::Point2f(direction_src[0], direction_src[1]);

    cv::Point2f dst[3] = {cv::Point2f(0,0), cv::Point2f(0,0), cv::Point2f(0,0)};
    dst[0] = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
    dst[1] = cv::Point2f(dst_w * 0.5 + dst_dir[0], dst_h * 0.5 + dst_dir[1]);

    float direction_dst[] = {0, 0};
    _get_3rd_point(dst[0], dst[1], direction_dst);
    dst[2] = cv::Point2f(direction_dst[0], direction_dst[1]);

    if (inv){
        trans = cv::getAffineTransform(dst, src);
    } else {
        trans = cv::getAffineTransform(src, dst);
    }
}


void pretty_print(const ncnn::Mat &m, std::vector<float> &vec_heap){
    for (int q=0; q < m.c; q++) {
        const float *ptr = m.channel(q);
        for (int z=0; z< m.d; z++) {
            for (int y=0; y < m.h; y++) {
                for (int x=0; x < m.w; x++) {
                    vec_heap.emplace_back(ptr[x]);
                }
                ptr += m.w;
            }

        }
    }
}

void flip_ncnn(const ncnn::Mat m, ncnn::Mat &in_flip) {

    for (int q=0; q < m.c; q++) {
        float *ptr = (float *) (ncnn::Mat) m.channel(q);
        for (int z =0; z < m.d; z++) {
            for (int y = 0; y < m.h; y++) {
                for (int x = 0; x < m.w /2 ; x++) {

                    float swap = ptr[x]; 
                    ptr[x] = ptr[m.w - x - 1];
                    ptr[m.w - x - 1] = swap;

                }
                ptr += m.w ;
            }
        }
    }
}

void flip_ncnn_mat(const ncnn::Mat m, ncnn::Mat &flip_in) {

    for (int q=0; q < m.c; q++) {
        const float *ptr = (float *) (ncnn::Mat) m.channel(q);
        ncnn::Mat flip_p = flip_in.channel(q);
        float *flip_ptr = (float *) flip_p;
        for (int z =0; z < m.d; z++) {
            for (int y = 0; y < m.h; y++) {
                for (int x = m.w; x >= 0 ; x--) {

                    flip_ptr[m.w - x - 1] = ptr[x];

                }
            ptr += m.w ;
            flip_ptr += m.w ;
            }
        }
    }
}


void
transform_preds(std::vector<cv::Point2f> coords, std::vector<Keypoints> &target_coords, float *center, float *scale,
                int w, int h, bool use_udp = false) {
    float scale_x[] = {0, 0};
    float temp_scale[] = {scale[0] * 200, scale[1] * 200};
    if (use_udp) {
        scale_x[0] = temp_scale[0] / (w - 1);
        scale_x[1] = temp_scale[1] / (h - 1);
    } else {
        scale_x[0] = temp_scale[0] / w;
        scale_x[1] = temp_scale[1] / h;
    }
    for (int i = 0; i < coords.size(); i++) {
        target_coords[i].x = coords[i].x * scale_x[0] + center[0] - temp_scale[0] * 0.5;
        target_coords[i].y = coords[i].y * scale_x[1] + center[1] - temp_scale[1] * 0.5;
    }
 
}
 
void pretty_exchange_channel(ncnn::Mat &flip_result, int flip_pairs[][2]) {
 
    for (int i = 0; i < sizeof(flip_pairs) / sizeof(flip_pairs[0]); i++) {
        ncnn::Mat q = flip_result.channel(flip_pairs[i][0]);
        flip_result.channel(flip_pairs[i][0]) = flip_result.channel(flip_pairs[i][1]);
        flip_result.channel(flip_pairs[i][1]) = q;
    }
}
 
void pretty_print(const ncnn::Mat &m) {
    for (int q = 0; q < m.c; q++) {
        const float *ptr = m.channel(q);
        for (int z = 0; z < m.d; z++) {
            for (int y = 0; y < m.h; y++) {
                for (int x = 0; x < m.w; x++) {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}

// int main() {
//     bool flip_test = true;
//     float image_target_w = 256;
//     float image_target_h = 192;
//     float output_size[] = {};
//     return 0;
// }


int main(int argc, char **argv) {

    bool flip_test = true;
    bool heap_map= false;
    float keypoint_score=0.3f;
    int flip_pairs[][2] = {{1,  2},{3,  4},{5,  6},{7,  8},
                             {9,  10},{11, 12},{13, 14},{15, 16}};
    cv::Mat bgr = cv::imread("../0.jpg");
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
 
    float image_target_w = 256;
    float image_target_h = 192;
    float padding = 1.25;
    float pixel_std = 200;
    float aspect_ratio = image_target_h / image_target_w;
    float bbox[] = {2.213932e+02, 1.935179e+02, 9.873443e+02, 1.035825e+03, 9.995332e-01};// 需要检测框架　这个矩形框来自检测框架的坐标　x y w h score
    bbox[2] = bbox[2] - bbox[0];
    bbox[3] = bbox[3] - bbox[1];
    float center[2] = {0, 0};
    float scale[2] = {0, 0};
    bbox_xywh2cs(bbox, aspect_ratio, padding, pixel_std, center, scale);
    float rot = 0;
    float shift[] = {0, 0};
    bool inv = false;
    float output_size[] = {image_target_h, image_target_w};
    cv::Mat trans;
    get_affine_transform(center, scale, rot, output_size, shift, inv, trans);
    std::cout << trans << std::endl;
    cv::Mat detect_image;//= cv::Mat::zeros(image_target_w ,image_target_h, CV_8UC3);
    cv::warpAffine(rgb, detect_image, trans, cv::Size(image_target_h, image_target_w), cv::INTER_LINEAR);
 
    std::cout << detect_image.cols << " " << detect_image.rows << std::endl;
    //std::cout << detect_image<<std::endl;
    ncnn::Net harnet;
 
    harnet.load_param("../model/tmp_sim.param");
    harnet.load_model("../model/tmp_sim.bin");
 
    ncnn::Mat in = ncnn::Mat::from_pixels(detect_image.data, ncnn::Mat::PIXEL_RGB, detect_image.cols,
                                          detect_image.rows);
    // transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_255[3] = {(1 / 0.229f / 255.f), (1 / 0.224f / 255.f), (1 / 0.225f / 255.f)};
    in.substract_mean_normalize(mean_vals, norm_255);
 
    fprintf(stderr, "input shape: %d %d %d %d\n", in.d, in.h, in.w, in.c);
 
    ncnn::Extractor ex = harnet.create_extractor();
 
    ex.input("input.1", in);//input 是 .param文件中输入节点名称
 
    ncnn::Mat result;
 
    ex.extract("2947", result);
    fprintf(stderr, "output shape: %d %d %d %d\n", result.d, result.c, result.h, result.w);
    int shape_c = result.c;
    int shape_w = result.w;
    int shape_h = result.h;
    std::vector<float> vec_heap;
    pretty_print(result, vec_heap);
    std::vector<float> vec_flip_heap;
    std::vector<float> vec_result_heap = vec_heap;
    if (flip_test) {
        ncnn::Mat flip_in;
        flip_in.create(in.w, in.h, in.d, in.c);
        //pretty_print(in);
        flip_ncnn_mat(in, flip_in);//flip(3)
        //pretty_print(flip_in);
        fprintf(stderr, "input shape: %d %d %d %d\n", flip_in.d, flip_in.c, flip_in.h, flip_in.w);
        ex.input("input.1", flip_in);//input 是 .param文件中输入节点名称
        ncnn::Mat flip_result;
        ex.extract("2947", flip_result);
        fprintf(stderr, "flip_output shape: %d %d %d %d\n", flip_result.d, flip_result.c, flip_result.h, flip_result.w);
        int flip_shape_c = flip_result.c;
        int flip_shape_w = flip_result.w;
        int flip_shape_h = flip_result.h;
 
        pretty_exchange_channel(flip_result, flip_pairs);
        pretty_print(flip_result, vec_flip_heap);
        #pragma omp parallel for num_threads(omp_get_max_threads())
        for (int i = 0; i < vec_result_heap.size(); i++) {
            vec_result_heap[i] = (vec_heap[i] + vec_flip_heap[i]) / 2;
        }
    }
 
    std::vector<Keypoints> all_preds;
    std::vector<int> idx;
    for (int i = 0; i < shape_c; i++) {
        auto begin = vec_result_heap.begin() + i * shape_w * shape_h;
        auto end = vec_result_heap.begin() + (i + 1) * shape_w * shape_h;
        float maxValue = *max_element(begin, end);
        int maxPosition = max_element(begin, end) - begin;
        all_preds.emplace_back(Keypoints(0, 0, maxValue));
        idx.emplace_back(maxPosition);
    }
    std::vector<cv::Point2f> vec_point;
    for (int i = 0; i < idx.size(); i++) {
        int x = idx[i] % shape_w;
        int y = idx[i] / shape_w;
        vec_point.emplace_back(cv::Point2f(x, y));
    }
 
 
    for (int i = 0; i < shape_c; i++) {
        int px = vec_point[i].x;
        int py = vec_point[i].y;
        if (px > 1 && px < shape_w - 1 && py > 1 && py < shape_h - 1) {
            float diff_0 = vec_heap[py * shape_w + px + 1] - vec_heap[py * shape_w + px - 1];
            float diff_1 = vec_heap[(py + 1) * shape_w + px] - vec_heap[(py - 1) * shape_w + px];
            vec_point[i].x += diff_0 == 0 ? 0 : (diff_0 > 0) ? 0.25 : -0.25;
            vec_point[i].y += diff_1 == 0 ? 0 : (diff_1 > 0) ? 0.25 : -0.25;
        }
    }
    std::vector<Box> all_boxes;
    if(heap_map){
        all_boxes.emplace_back(Box(center[0], center[1], scale[0], scale[1], scale[0] * scale[1] * 400, bbox[4]));
    }
    transform_preds(vec_point, all_preds, center, scale, shape_w, shape_h);
    int skeleton[][2] = {{15, 13},{13, 11},{16, 14},{14, 12},
                         {11, 12},{5,  11},{6,  12},{5,  6},
                         {5,  7},{6,  8},{7,  9},{8,  10},
                         {1,  2},{0,  1},{0,  2},{1,  3},
                         {2,  4},{3,  5},{4,  6}};
 
    cv::rectangle(bgr, cv::Point(bbox[0], bbox[1]), cv::Point(bbox[0] + bbox[2], bbox[1] + bbox[3]),
                  cv::Scalar(255, 0, 0));
    for (int i = 0; i < all_preds.size(); i++) {
        if(all_preds[i].score>keypoint_score){
            cv::circle(bgr, cv::Point(all_preds[i].x, all_preds[i].y), 3, cv::Scalar(0, 255, 120), -1);//画点，其实就是实心圆
        }
    }
    for (int i = 0; i < sizeof(skeleton) / sizeof(sizeof(skeleton[1])); i++) {
        int x0 = all_preds[skeleton[i][0]].x;
        int y0 = all_preds[skeleton[i][0]].y;
        int x1 = all_preds[skeleton[i][1]].x;
        int y1 = all_preds[skeleton[i][1]].y;
 
        cv::line(bgr, cv::Point(x0, y0), cv::Point(x1, y1),
                 cv::Scalar(0, 255, 0), 1);
 
    }
    cv::imshow("image", bgr);
    cv::waitKey(0);
    return 0;
}