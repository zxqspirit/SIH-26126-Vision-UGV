from setuptools import find_packages, setup

package_name = 'sih_visualization'

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
    maintainer='Member 4 - Integration Engineer',
    maintainer_email='harshpatel05072005@gmail.com',
    description='Telemetry aggregator and visualization bridge for dashboard and RViz2 for SIH 26126 UGV',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'telemetry_bridge_node = sih_visualization.telemetry_bridge_node:main',
        ],
    },
)
