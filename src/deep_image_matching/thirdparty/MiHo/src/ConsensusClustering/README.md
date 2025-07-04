# Multi Consensus Clustering

The algorithm proposed in paper: Daniel Barath, Denys Rozumnyi, Ivan Eichhardt, Levente Hajder, Jiri Matas; Finding Geometric Models by Clustering in the Consensus Space, Conference on Computer Vision and Pattern Recognition, 2023. 

# Installation C++

To build and install the C++ version, clone or download this repository and then build the project by CMAKE. 
```shell
$ git clone --recursive https://github.com/danini/clustering-in-consensus-space.git
$ cd build
$ cmake ..
$ make
```

# Install Python package and compile C++

```bash
python3 ./setup.py install
```

or

```bash
pip3 install -e .
```

# Example project

To build the sample project showing examples of two-view motion and homography fitting, set variable `CREATE_SAMPLE_PROJECT = ON` when creating the project in CMAKE. 
Then 
```shell
$ cd build
$ ./SampleProject
```

# Jupyter Notebook code for re-producing the results in the paper

The code for multiple homography fitting is available at: [notebook](dataset_comparison/adelaideH.ipynb).

The code for multiple two-view motion fitting is available at: [notebook](dataset_comparison/adelaideF.ipynb).

# Jupyter Notebook example

The example for multiple homography fitting is available at: [notebook](examples/example_multi_homography.ipynb).

The example for multiple two-view motion fitting is available at: [notebook](examples/example_multi_two_view_motion.ipynb).
 
# Requirements

- Eigen 3.0 or higher
- CMake 2.8.12 or higher
- OpenCV 3.0 or higher
- GFlags
- GLog
- A modern compiler with C++17 support


# Acknowledgements

When using the algorithm, please cite `Daniel Barath, Denys Rozumnyi, Ivan Eichhardt, Levente Hajder, Jiri Matas. "Finding Geometric Models by Clustering in the Consensus Space, Conference on Computer Vision and Pattern Recognition". 2023`.

If you use Progressive-X with Progressive NAPSAC sampler, please cite `Barath, Daniel and Noskova, Jana and Ivashechkin, Maksym and Matas, Jiří. "MAGSAC++, a fast, reliable and accurate robust estimator" Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2020`.
