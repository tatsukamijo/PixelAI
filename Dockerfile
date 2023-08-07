FROM ubuntu:16.04

# Add ROS repository and keys
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list' \
    && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Update package lists
RUN apt-get update

# Install necessary dependencies
RUN apt-get install -y \
    curl wget git vim python2.7 python-pip \
    # ros-kinetic-desktop-full \
    && apt-get remove -y ros-kinetic-desktop-full ros-kinetic-gazebo* libgazebo* gazebo* \
    && rm -rf /var/lib/apt/lists/*

# Install Gazebo 
RUN curl -sSL http://get.gazebosim.org | sh

RUN apt-get remove -y ros-kinetic-desktop-full ros-kinetic-gazebo* libgazebo* gazebo* \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for Python 2.7 and install Pytorch and torchvision
RUN pip install --upgrade "pip < 21.0" \
    && pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl \
    torchvision==0.2.1

# Set the working directory
WORKDIR /PixelAI

# Source ROS and workspace
RUN echo "source /opt/ros/kinetic/setup.sh" >> ~/.bashrc \
    && echo "source /PixelAI/nao_simulation/catkin_ws/devel/setup.bash" >> ~/.bashrc \
    && echo "export PS1=\"\[\e[1;34m\][\u@\h \W]\\$ \[\e[m\]\" " >> ~/.bashrc

# Upgrade system and install ROS packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
    ros-kinetic-ros-base \
    ros-kinetic-gazebo9-ros-pkgs ros-kinetic-gazebo9-ros-control ros-kinetic-gazebo9* \
    ros-kinetic-catkin rviz \
    ros-kinetic-controller-manager ros-kinetic-joint-state-controller ros-kinetic-joint-trajectory-controller ros-kinetic-rqt ros-kinetic-rqt-controller-manager ros-kinetic-rqt-joint-trajectory-controller ros-kinetic-ros-control ros-kinetic-rqt-gui \
    ros-kinetic-rqt-plot ros-kinetic-rqt-graph ros-kinetic-rqt-rviz ros-kinetic-rqt-tf-tree \
    ros-kinetic-gazebo9-ros ros-kinetic-kdl-conversions ros-kinetic-kdl-parser ros-kinetic-forward-command-controller ros-kinetic-tf-conversions ros-kinetic-xacro ros-kinetic-joint-state-publisher ros-kinetic-robot-state-publisher \
    ros-kinetic-ros-control ros-kinetic-ros-controllers

# Install catkin tools
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list' \
    && wget http://packages.ros.org/ros.key -O - | sudo apt-key add - \
    && apt-get -y update \
    && apt-get -y install python3-catkin-tools

