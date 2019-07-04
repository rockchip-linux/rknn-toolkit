cat <<EOF > "91-rk1808-ai-cs.rules"
SUBSYSTEM=="usb", ATTR{idVendor}=="2207", MODE="0666"
EOF

sudo cp -f 91-rk1808-ai-cs.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo ldconfig
rm 91-rk1808-ai-cs.rules
