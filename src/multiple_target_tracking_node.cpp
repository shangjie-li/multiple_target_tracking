#include <omp.h>
#include <ros/ros.h>
#include <string>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Header.h>

#include <Eigen/Dense>
#include <cassert>
#include <limits.h>
#include <float.h>

#include "kalman.hpp"
#include "obj.hpp"

static std::vector<int> COLOR_B = {244, 233, 156, 103, 63, 33, 3, 0, 0};
static std::vector<int> COLOR_G = {67, 30, 99, 58, 81, 150, 169, 188, 150};
static std::vector<int> COLOR_R = {54, 99, 176, 183, 181, 243, 244, 212, 136};

class MTT
{
public:
    MTT(ros::NodeHandle &nh);
    ~MTT();

private:
    std::string sub_topic_;
    std::string pub_topic_;
    std::string frame_id_;

    bool show_objects_num_;
    bool show_time_;

    double time_interval_;
    double gate_threshold_;
    int blind_update_limit_;

    double sigma_ax_;
    double sigma_ay_;
    double sigma_ox_;
    double sigma_oy_;

    double min_scale_;
    double max_scale_;
    double min_height_;
    double max_height_;

    ros::Subscriber sub_;
    ros::Publisher pub_;

    std::vector<Object> objs_;
    std::vector<Object> objs_temp_;
    int number_;

    void get_observation(const visualization_msgs::MarkerArray *markers_in,
                          std::vector<Object> *objs_observed,
                          std::vector<Object> *objs_ignored);

    void manage_objs(std::vector<Object> *objs_observed,
                        std::vector<Object> *objs,
                        std::vector<Object> *objs_temp);
    
    void delete_objs(std::vector<Object> *objs);

    void augment_objs(std::vector<Object> *objs_observed,
                      std::vector<Object> *objs,
                      std::vector<Object> *objs_temp);

    void init_objs(std::vector<Object> *objs_observed,
                      std::vector<Object> *objs_temp);

    void publish_markers(ros::Publisher &pub,
                  visualization_msgs::MarkerArray *markers_out,
                  std::vector<Object> *objs,
                  std::vector<Object> *objs_ignored,
                  std_msgs::Header &header);
    
    void callback(const visualization_msgs::MarkerArray &markers_in);
};

MTT::MTT(ros::NodeHandle &nh)
{
    nh.param<std::string>("sub_topic", sub_topic_, "/objects");
    nh.param<std::string>("pub_topic", pub_topic_, "/objects_tracked");
    nh.param<std::string>("frame_id", frame_id_, "pandar");

    nh.param<bool>("show_objects_num", show_objects_num_, false);
    nh.param<bool>("show_time", show_time_, false);

    nh.param<double>("time_interval", time_interval_, 0.1);
    nh.param<double>("gate_threshold", gate_threshold_, 1000);
    nh.param<int>("blind_update_limit", blind_update_limit_, 5);

    nh.param<double>("sigma_ax", sigma_ax_, 0.1);
    nh.param<double>("sigma_ay", sigma_ay_, 0.1);
    nh.param<double>("sigma_ox", sigma_ox_, 0.1);
    nh.param<double>("sigma_oy", sigma_oy_, 0.1);

    nh.param<double>("min_scale", min_scale_, 1.0);
    nh.param<double>("max_scale", max_scale_, 6.0);
    nh.param<double>("min_height", min_height_, 1.0);
    nh.param<double>("max_height", max_height_, 2.5);

    sub_ = nh.subscribe(sub_topic_, 1, &MTT::callback, this);
    pub_ = nh.advertise<visualization_msgs::MarkerArray>(pub_topic_, 1);

    number_ = 0;

    ros::spin();
}

MTT::~MTT() {}

void MTT::get_observation(const visualization_msgs::MarkerArray *markers_in,
                          std::vector<Object> *objs_observed,
                          std::vector<Object> *objs_ignored)
{
    for (int i = 0; i < markers_in->markers.size(); i ++)
    {
        Object obj;
        obj.x0 = markers_in->markers[i].pose.position.x;
        obj.y0 = markers_in->markers[i].pose.position.y;
        obj.z0 = markers_in->markers[i].pose.position.z;
        obj.l = markers_in->markers[i].scale.x;
        obj.w = markers_in->markers[i].scale.y;
        obj.h = markers_in->markers[i].scale.z;

        double xx = markers_in->markers[i].pose.orientation.x;
        double yy = markers_in->markers[i].pose.orientation.y;
        double zz = markers_in->markers[i].pose.orientation.z;
        double ww = markers_in->markers[i].pose.orientation.w;
        Eigen::Quaterniond q(ww, xx, yy, zz);
        Eigen::Vector3d q_eul = q.toRotationMatrix().eulerAngles(2, 1, 0);
        double phi = q_eul[0];

        obj.phi = phi;
        obj.has_orientation = false;

        obj.xref = markers_in->markers[i].pose.position.x;
        obj.yref = markers_in->markers[i].pose.position.y;

        std::vector<double> scales = {(double)obj.l, (double)obj.w};
        std::sort(scales.begin(), scales.end());
        double obj_scale = scales[scales.size() - 1];
        double obj_height = (double)obj.h;
        
        if ((obj_scale >= min_scale_) && (obj_scale <= max_scale_) && (obj_height >= min_height_) && (obj_height <= max_height_))
        {
            objs_observed->push_back(obj);
        }
        else
        {
            objs_ignored->push_back(obj);
        }
        
    }
}

void MTT::manage_objs(std::vector<Object> *objs_observed,
                      std::vector<Object> *objs,
                      std::vector<Object> *objs_temp)
{
    #pragma omp for
    for (int j = 0; j < objs->size(); j ++)
    {
        bool flag = false;
        int idx = 0;
        double ddm = DBL_MAX;

        for (int k = 0; k < objs_observed->size(); k ++)
        {
            double x = (*objs_observed)[k].xref;
            double y = (*objs_observed)[k].yref;
            double dd = (*objs)[j].tracker.compute_the_residual(x, y);
            if ((dd < ddm) && (dd < gate_threshold_)) {idx = k; ddm = dd; flag = true;}
        }

        if (flag)
        {
            double zx = (*objs_observed)[idx].xref;
            double zy = (*objs_observed)[idx].yref;
            (*objs)[j].tracker.predict();
            (*objs)[j].tracker.update(zx, zy);
            (*objs)[j].tracker_blind_update = 0;

            (*objs)[j].xref = (*objs)[j].tracker.get_state()(0, 0);
            (*objs)[j].vx = (*objs)[j].tracker.get_state()(1, 0);
            (*objs)[j].yref = (*objs)[j].tracker.get_state()(2, 0);
            (*objs)[j].vy = (*objs)[j].tracker.get_state()(3, 0);

            (*objs)[j].x0 = (*objs_observed)[idx].x0;
            (*objs)[j].y0 = (*objs_observed)[idx].y0;
            (*objs)[j].z0 = (*objs_observed)[idx].z0;
            (*objs)[j].l = (*objs_observed)[idx].l;
            (*objs)[j].w = (*objs_observed)[idx].w;
            (*objs)[j].h = (*objs_observed)[idx].h;
            (*objs)[j].phi = (*objs_observed)[idx].phi;
            (*objs)[j].has_orientation = (*objs_observed)[idx].has_orientation;

            objs_observed->erase(objs_observed->begin() + idx);
        }
        else
        {
            (*objs)[j].tracker.predict();
            (*objs)[j].tracker_blind_update += 1;

            (*objs)[j].xref = (*objs)[j].tracker.get_state()(0, 0);
            (*objs)[j].vx = (*objs)[j].tracker.get_state()(1, 0);
            (*objs)[j].yref = (*objs)[j].tracker.get_state()(2, 0);
            (*objs)[j].vy = (*objs)[j].tracker.get_state()(3, 0);
            (*objs)[j].x0 = (*objs)[j].xref;
            (*objs)[j].y0 = (*objs)[j].yref;
        }
    }
}

void MTT::delete_objs(std::vector<Object> *objs)
{
    int test_k = 0;
    while (test_k < objs->size())
    {
        if ((*objs)[test_k].tracker_blind_update > blind_update_limit_) {objs->erase(objs->begin() + test_k); continue;}
        test_k += 1;
    }
}

void MTT::augment_objs(std::vector<Object> *objs_observed,
                      std::vector<Object> *objs,
                      std::vector<Object> *objs_temp)
{
    #pragma omp for
    for (int j = 0; j< objs_temp->size(); j ++)
    {
        bool flag = false;
        int idx = 0;
        double ddm = DBL_MAX;

        for (int k = 0; k < objs_observed->size(); k ++)
        {
            double x = (*objs_observed)[k].xref;
            double y = (*objs_observed)[k].yref;
            double dd = (*objs_temp)[j].tracker.compute_the_residual(x, y);
            if ((dd < ddm) && (dd < gate_threshold_)) {idx = k; ddm = dd; flag = true;}
        }

        if (flag)
        {
            double zx = (*objs_observed)[idx].xref;
            double zy = (*objs_observed)[idx].yref;
            double x = (*objs_temp)[j].tracker.get_state()(0, 0);
            double y = (*objs_temp)[j].tracker.get_state()(2, 0);
            double t = time_interval_;
            (*objs_temp)[j].tracker.init(t, zx, (zx - x) / t, zy, (zy - y) / t, sigma_ax_, sigma_ay_, sigma_ox_, sigma_oy_);

            objs_observed->erase(objs_observed->begin() + idx);

            number_ += 1;
            number_ = number_ % 1000;
            (*objs_temp)[j].number = number_;
            (*objs_temp)[j].x0 = (*objs_observed)[idx].x0;
            (*objs_temp)[j].y0 = (*objs_observed)[idx].y0;
            (*objs_temp)[j].z0 = (*objs_observed)[idx].z0;
            (*objs_temp)[j].l = (*objs_observed)[idx].l;
            (*objs_temp)[j].w = (*objs_observed)[idx].w;
            (*objs_temp)[j].h = (*objs_observed)[idx].h;
            (*objs_temp)[j].phi = (*objs_observed)[idx].phi;
            (*objs_temp)[j].has_orientation = (*objs_observed)[idx].has_orientation;

            assert((COLOR_B.size() == COLOR_G.size()) && (COLOR_B.size() == COLOR_R.size()));
            int num_c = COLOR_R.size();
            (*objs_temp)[j].color_r = COLOR_R[number_ % num_c];
            (*objs_temp)[j].color_g = COLOR_G[number_ % num_c];
            (*objs_temp)[j].color_b = COLOR_B[number_ % num_c];
            objs->push_back((*objs_temp)[j]);
        }
    }
}

void MTT::init_objs(std::vector<Object> *objs_observed,
                      std::vector<Object> *objs_temp)
{
    *objs_temp = *objs_observed;
    #pragma omp for
    for (int j = 0; j < objs_temp->size(); j ++)
    {
        double x = (*objs_temp)[j].xref;
        double y = (*objs_temp)[j].yref;
        (*objs_temp)[j].tracker.init(time_interval_, x, 0, y, 0, sigma_ax_, sigma_ay_, sigma_ox_, sigma_oy_);
    }
    objs_observed->clear();
}

void MTT::publish_markers(ros::Publisher &pub,
                  visualization_msgs::MarkerArray *markers_out,
                  std::vector<Object> *objs,
                  std::vector<Object> *objs_ignored,
                  std_msgs::Header &header)
{
    for (int i = 0; i < objs->size(); i ++)
    {
        visualization_msgs::Marker mar;
        mar.header = header;

        // 设置该标记的命名空间和ID，ID应该是独一无二的
        // 具有相同命名空间和ID的标记将会覆盖
        mar.ns = "obstacle";
        mar.id = (*objs)[i].number;

        // 设置标记类型
        mar.type = visualization_msgs::Marker::CUBE;
        
        // 设置标记行为，ADD为添加，DELETE为删除
        mar.action = visualization_msgs::Marker::ADD;

        // 设置标记位姿
        mar.pose.position.x = (*objs)[i].x0;
        mar.pose.position.y = (*objs)[i].y0;
        mar.pose.position.z = (*objs)[i].z0;
        mar.pose.orientation.x = 0;
        mar.pose.orientation.y = 0;
        mar.pose.orientation.z = sin(0.5 * (*objs)[i].phi);
        mar.pose.orientation.w = cos(0.5 * (*objs)[i].phi);

        // 设置标记尺寸
        mar.scale.x = (*objs)[i].l;
        mar.scale.y = (*objs)[i].w;
        mar.scale.z = (*objs)[i].h;

        // 设置标记颜色，应确保不透明度alpha非零
        mar.color.r = (float)(*objs)[i].color_r / 255;
        mar.color.g = (float)(*objs)[i].color_g / 255;
        mar.color.b = (float)(*objs)[i].color_b / 255;
        mar.color.a = 0.85;

        // 设置标记生存时间，单位为s
        mar.lifetime = ros::Duration(0.1);
        mar.text = ' ';

        markers_out->markers.push_back(mar);
    }

    for (int i = 0; i < objs_ignored->size(); i ++)
    {
        visualization_msgs::Marker mar;
        mar.header = header;

        // 设置该标记的命名空间和ID，ID应该是独一无二的
        // 具有相同命名空间和ID的标记将会覆盖
        mar.ns = "obstacle";
        mar.id = i % 1000 + 1000;

        // 设置标记类型
        mar.type = visualization_msgs::Marker::CUBE;
        
        // 设置标记行为，ADD为添加，DELETE为删除
        mar.action = visualization_msgs::Marker::ADD;

        // 设置标记位姿
        mar.pose.position.x = (*objs_ignored)[i].x0;
        mar.pose.position.y = (*objs_ignored)[i].y0;
        mar.pose.position.z = (*objs_ignored)[i].z0;
        mar.pose.orientation.x = 0;
        mar.pose.orientation.y = 0;
        mar.pose.orientation.z = sin(0.5 * (*objs_ignored)[i].phi);
        mar.pose.orientation.w = cos(0.5 * (*objs_ignored)[i].phi);

        // 设置标记尺寸
        mar.scale.x = (*objs_ignored)[i].l;
        mar.scale.y = (*objs_ignored)[i].w;
        mar.scale.z = (*objs_ignored)[i].h;

        // 设置标记颜色，应确保不透明度alpha非零
        mar.color.r = 0.5f;
        mar.color.g = 0.5f;
        mar.color.b = 0.5f;
        mar.color.a = 0.85;

        // 设置标记生存时间，单位为s
        mar.lifetime = ros::Duration(0.1);
        mar.text = ' ';

        markers_out->markers.push_back(mar);
    }

    pub.publish(*markers_out);
}

void MTT::callback(const visualization_msgs::MarkerArray &markers_in)
{
    ros::Time time_start = ros::Time::now();

    // 获取量测
    std::vector<Object> objs_observed;
    std::vector<Object> objs_ignored;
    get_observation(&markers_in, &objs_observed, &objs_ignored);

    // 数据关联与状态更新
    manage_objs(&objs_observed, &objs_, &objs_temp_);

    // 删除长时间未关联的目标
    delete_objs(&objs_);

    // 增广跟踪列表
    augment_objs(&objs_observed, &objs_, &objs_temp_);

    // 增广临时跟踪列表
    init_objs(&objs_observed, &objs_temp_);

    visualization_msgs::MarkerArray markers_out;
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = frame_id_;
    publish_markers(pub_, &markers_out, &objs_, &objs_ignored, header);

    ros::Time time_end = ros::Time::now();

    if (show_objects_num_ || show_time_)
    {
        std::cout << std::endl;
    }

    if (show_objects_num_)
    {
        std::cout << "size of observed objects:   " << markers_in.markers.size() << std::endl;
        std::cout << "size of tracked objects:    " << objs_.size() << std::endl;
        std::cout << "size of initialized objects:" << objs_temp_.size() << std::endl;
        std::cout << "size of markers:            " << markers_out.markers.size() << std::endl;
    }

    if (show_time_)
    {
        std::cout << "cost time:" << time_end - time_start << "s" << std::endl;
    }

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "multiple_target_tracking");
    ros::NodeHandle nh("~");

    omp_set_num_threads(4);

    MTT mtt(nh);
    return 0;
}