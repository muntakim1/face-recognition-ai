from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="face-recognition-ai",
    version="0.0.1",
    description="A face recognition system using yoloface and facenet",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/muntakim1/face-recognition-ai",
    author="Muntakimur Rahaman",
    author_email="muntakim.cse@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "facenet-pytorch",
        "PyYaml",
        "joblib",
        "opencv-python",
        "yolo5face",
    ],
    extra_require={"dev": ["twine"]},
    python_requires=">=3.10",
)
