from setuptools import setup
import re


def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


setup(
    name="igps_kdar",
    version=get_property("__version__", "igps_kdar"),
    description="IGPS HI Absorption Spectrum KDAR Determination",
    author="Trey V. Wenger",
    packages=["igps_kdar"],
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "astropy",
        "scikit-learn",
        "tensorflow[and-cuda]",
    ],
    entry_points={
        "console_scripts": [
            "igps-kdar-generate-spectra=igps_kdar.generate_spectra:main",
            "igps-kdar-classify-jones=igps_kdar.classify_jones:main",
            "igps-kdar-classify-kolpak=igps_kdar.classify_kolpak:main",
            "igps-kdar-generate-labels=igps_kdar.generate_labels:main",
            "igps-kdar-simulate-spectra=igps_kdar.simulate_spectra:main",
            "igps-kdar-train-cnn=igps_kdar.train_cnn:main",
            "igps-kdar-hypertrain-cnn=igps_kdar.hyper_train_cnn:main",
            "igps-kdar-test-cnn=igps_kdar.test_cnn:main",
        ]
    },
)
