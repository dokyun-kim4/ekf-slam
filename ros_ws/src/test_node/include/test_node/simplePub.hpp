#pragma once
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>
#include <chrono>

using namespace std::chrono_literals;

class SimplePublisher : public rclcpp::Node
{
    public:
        SimplePublisher(const std::string &name) : Node(name)
        {
            publisher_ = this->create_publisher<std_msgs::msg::String>("test_topic", 10);
            timer_ = this->create_wall_timer(500ms, std::bind(&SimplePublisher::timer_callback, this));
        }

    private:
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
        rclcpp::TimerBase::SharedPtr timer_;

        void timer_callback()
        {
            auto message = std_msgs::msg::String();
            message.data = "Hello, ROS 2!";
            RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
            publisher_->publish(message);
        }
};