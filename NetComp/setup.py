from setuptools import setup

setup(name='NetComp',
      version='0.2.3',
      description='Network Comparison',
      license='MIT',
      author='Peter Wills',
      author_email='peter.e.wills@gmail.com',
      packages=[
          'netcomp',
          'netcomp.linalg',
          'netcomp.distance'
      ],
      url='https://github.com/peterewills/NetComp',
      install_requires=[
          'numpy>=1.11.3',
          'scipy>=0.18',
          'networkx<2'
      ]
     )
