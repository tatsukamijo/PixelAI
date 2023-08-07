FROM ubuntu:16.04

# Add ROS repository
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Update package lists
RUN apt-get update

# Install necessary dependencies
RUN apt-get install -y \
    curl \
    wget \
    git \
    vim \
    python2.7 \
    python-pip \
    ros-kinetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

# Uninstall Gazebo 7
RUN apt-get remove -y ros-kinetic-gazebo* libgazebo* gazebo*

# Install Gazebo 9/10?
RUN curl -sSL http://get.gazebosim.org | sh

# Upgrade pip for Python 2.7
RUN pip install --upgrade "pip < 21.0"

# Install Pytorch 0.4.1 and torchvision 0.2.1 for Python 2.7
RUN pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl \
    && pip install torchvision==0.2.1

# Set the working directory to the cloned repository
WORKDIR /PixelAI

# Clone the PixelAI repository
# Add any additional steps required for your specific project

# Source ROS and workspace
RUN echo "source /opt/ros/kinetic/setup.sh" >> ~/.bashrc
RUN echo "source /PixelAI/nao_simulation/catkin_ws/devel/setup.bash" >> ~/.bashrc
RUN echo "export PS1="\[\e[1;34m\][\u@\h \W]\\$ \[\e[m\]" " >> ~/.bashrc

RUN apt-get remove ros-kinetic-desktop-full
RUN apt-get remove ros-kinetic-gazebo*
RUN apt-get remove gazebo* -y
RUN apt-get upgrade -y
# # Install Gazebo 9
RUN apt-get update
RUN apt-get install ros-kinetic-ros-base
RUN apt-get install -y ros-kinetic-gazebo9-ros-pkgs ros-kinetic-gazebo9-ros-control ros-kinetic-gazebo9* 
RUN apt-get install -y ros-kinetic-catkin
RUN apt-get install -y rviz
RUN apt-get install -y ros-kinetic-controller-manager ros-kinetic-joint-state-controller ros-kinetic-joint-trajectory-controller ros-kinetic-rqt ros-kinetic-rqt-controller-manager ros-kinetic-rqt-joint-trajectory-controller ros-kinetic-ros-control ros-kinetic-rqt-gui
RUN apt-get install -y ros-kinetic-rqt-plot ros-kinetic-rqt-graph ros-kinetic-rqt-rviz ros-kinetic-rqt-tf-tree
RUN apt-get install -y ros-kinetic-gazebo9-ros ros-kinetic-kdl-conversions ros-kinetic-kdl-parser ros-kinetic-forward-command-controller ros-kinetic-tf-conversions ros-kinetic-xacro ros-kinetic-joint-state-publisher ros-kinetic-robot-state-publisher
RUN apt-get install -y ros-kinetic-ros-control ros-kinetic-ros-controllers

# install catkin tools
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'
RUN wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
RUN apt-get update
RUN apt-get install python3-catkin-tools