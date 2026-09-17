from setuptools import find_packages, setup

package_name = 'sih_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Member 1 - Perception Engineer',
    maintainer_email='harshpatel05072005@gmail.com',
    description='Lightweight CNN terrain traversability classification for SIH 26126 UGV',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'traversability_classifier_node = sih_perception.traversability_classifier_node:main',
            'sensor_feed_publisher_node = sih_perception.sensor_feed_publisher_node:main',
        ],
    },
)
