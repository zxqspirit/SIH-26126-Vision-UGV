from setuptools import find_packages, setup

package_name = 'sih_navigation'

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
    maintainer='Member 3 - Planning Engineer',
    maintainer_email='harshpatel05072005@gmail.com',
    description='Local path planning and trajectory generation across traversability costmaps for SIH 26126 UGV',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'local_planner_node = sih_navigation.local_planner_node:main',
        ],
    },
)
