#include "simplePub.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimplePublisher>("simple_publisher");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}