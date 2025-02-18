from setuptools import setup

descr = """structure implement with numba"""

if __name__ == '__main__':
    setup(name='structron',
        version='0.0',
        url='https://github.com/VectorElectron/structron',
        description='queue, stack, heap, heash, avl, redblack...',
        long_description=descr,
        author='YXDragon',
        author_email='yxdragon@imagepy.org',
        license='BSD 3-clause',
        packages=['structron'],
        package_data={},
        install_requires=[
            'numba',
        ],
    )
