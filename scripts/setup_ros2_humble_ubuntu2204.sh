#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Please run this script as a normal user with sudo access, not as root."
  exit 1
fi

if [[ ! -f /etc/os-release ]]; then
  echo "/etc/os-release not found."
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release

if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "This script is intended for Ubuntu 22.04. Detected: ${PRETTY_NAME:-unknown}"
  exit 1
fi

echo "[1/7] Installing base packages..."
sudo apt update
sudo apt install -y curl gnupg2 lsb-release locales software-properties-common ca-certificates

echo "[2/7] Configuring locale..."
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "[3/7] Installing ROS 2 apt source package..."
ROS_APT_SOURCE_VERSION="$(
  curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest |
  grep -F '"tag_name"' |
  awk -F'"' '{print $4}'
)"

if [[ -z "${ROS_APT_SOURCE_VERSION}" ]]; then
  echo "Failed to fetch latest ros-apt-source version."
  exit 1
fi

UBUNTU_CODENAME_VALUE="${UBUNTU_CODENAME:-${VERSION_CODENAME:-jammy}}"
DEB_PATH="/tmp/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME_VALUE}_all.deb"

curl -L -o "${DEB_PATH}" \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME_VALUE}_all.deb"
sudo dpkg -i "${DEB_PATH}"

echo "[4/7] Updating apt indexes..."
sudo apt update

echo "[5/7] Installing ROS 2 Humble..."
sudo apt install -y ros-humble-ros-base ros-dev-tools

echo "[6/7] Adding ROS setup to ~/.bashrc if missing..."
if ! grep -Fq "source /opt/ros/humble/setup.bash" "${HOME}/.bashrc"; then
  echo "source /opt/ros/humble/setup.bash" >> "${HOME}/.bashrc"
fi

echo "[7/7] Verifying installation..."
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
which ros2
ros2 --help >/dev/null

echo "ROS 2 Humble installation finished."
echo "Open a new shell or run: source /opt/ros/humble/setup.bash"
