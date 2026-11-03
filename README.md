FaceFusion
==========

> Industry leading face manipulation platform.

[![Build Status](https://img.shields.io/github/actions/workflow/status/facefusion/facefusion/ci.yml.svg?branch=master)](https://github.com/facefusion/facefusion/actions?query=workflow:ci)
[![Coverage Status](https://img.shields.io/coveralls/facefusion/facefusion.svg)](https://coveralls.io/r/facefusion/facefusion)
![License](https://img.shields.io/badge/license-OpenRAIL--AS-green)


Preview
-------

![Preview](https://raw.githubusercontent.com/facefusion/facefusion/master/.github/preview.png?sanitize=true)


Installation
------------

Be aware, the [installation](https://docs.facefusion.io/installation) needs technical skills and is not recommended for beginners. In case you are not comfortable using a terminal, our [Windows Installer](http://windows-installer.facefusion.io) and [macOS Installer](http://macos-installer.facefusion.io) get you started.


macOS Acceleration
------------------

The standard macOS ONNX Runtime package includes the CoreML execution provider. FaceFusion automatically prefers CoreML when it is available, and configures it for fast prediction with a model cache. On Apple Silicon, CoreML can use the CPU, GPU, and Neural Engine; operators it cannot run may fall back to the CPU. CoreML requires macOS 10.15 or newer.

Confirm that the installed runtime includes CoreML:

```
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The output should include `CoreMLExecutionProvider`. To select it explicitly when running FaceFusion:

```
python facefusion.py run --execution-providers coreml
```

For a one-command launch, use `./run_mac.sh`. It opens the UI in your browser, uses the active Python environment if it has CoreML and the FaceFusion dependencies, and otherwise falls back to `uv` with the pinned requirements and Python 3.12. The first launch may install Python packages and download model files. Install `curl`, `ffmpeg`, and `ffprobe` first. Additional FaceFusion options can be passed through the script.


Usage
-----

Run the command:

```
python facefusion.py [commands] [options]

options:
  -h, --help                                      show this help message and exit
  -v, --version                                   show program's version number and exit

commands:
    run                                           run the program
    headless-run                                  run the program in headless mode
    batch-run                                     run the program in batch mode
    force-download                                force automate downloads and exit
    benchmark                                     benchmark the program
    job-list                                      list jobs by status
    job-create                                    create a drafted job
    job-submit                                    submit a drafted job to become a queued job
    job-submit-all                                submit all drafted jobs to become a queued jobs
    job-delete                                    delete a drafted, queued, failed or completed job
    job-delete-all                                delete all drafted, queued, failed and completed jobs
    job-add-step                                  add a step to a drafted job
    job-remix-step                                remix a previous step from a drafted job
    job-insert-step                               insert a step to a drafted job
    job-remove-step                               remove a step from a drafted job
    job-run                                       run a queued job
    job-run-all                                   run all queued jobs
    job-retry                                     retry a failed job
    job-retry-all                                 retry all failed jobs
```


Documentation
-------------

Read the [documentation](https://docs.facefusion.io) for a deep dive.
