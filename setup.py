from setuptools import setup

package_name = 'cv_basics'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sendol',
    maintainer_email='sendol@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'img_publisher = cv_basics.webcam_pub:main',
        'img_subscriber = cv_basics.webcam_sub:main',
        'realsense_publisher = cv_basics.realsense_pub:main',
        'pointcloud_publisher = cv_basics.pointcloud_pub:main',
        'pointcloud_subscriber = cv_basics.pointcloud_sub:main',
        'keyboard_op = cv_basics.keyboard_op:main'
    ],
},
)
