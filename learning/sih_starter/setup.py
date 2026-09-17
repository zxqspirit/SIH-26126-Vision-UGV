import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sih_starter'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SIH 26126 Team',
    maintainer_email='zxqspirit@todo.todo',
    description='Beginner ROS 2 educational package for SIH 26126 team',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'beacon_publisher = sih_starter.beacon_publisher:main',
            'beacon_subscriber = sih_starter.beacon_subscriber:main',
        ],
    },
)
