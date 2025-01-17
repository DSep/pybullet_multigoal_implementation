from setuptools import setup, find_packages


packages = find_packages()
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'drl_implementation' or p.startswith('drl_implementation.')

setup(name='drl-implementation',
      version='1.0.0',
      description='A collection of deep reinforcement learning algorithms for fast implementation. Based on XintongYang\'s implementation.',
      url='https://github.com/DSep/pybullet_multigoal_implementation',
      author='ddd26, mbg34, nyl25, sd974 and XintongYang',
      author_email='author@cam.ac.uk',
      packages=packages,
      package_dir={'drl_implementation': 'drl_implementation'},
      package_data={'drl_implementation': [
          'examples/*.md',
      ]},
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ])
