from setuptools import find_packages, setup

package_name = 'sih_localization'

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
    maintainer='Member 2 - Localization Engineer',
    maintainer_email='harshpatel05072005@gmail.com',
    description='Visual odometry and TF2 coordinate frame management for SIH 26126 UGV',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visual_odometry_node = sih_localization.visual_odometry_node:main',
        ],
    },
)
