# Mine Agent Final Project
Our src folder contains all of our relevant code. The engine.py is our main way of parsing arguments that are passed to our bash script - which we use to run our various experiments. Each of the other files has the main code for the various algorithms that we used for the given task: iq.py - the inverse q learning code that follows the example code from the paper, a2c.py - the a2c algorithm for our use case with the ability to load a pretrained model, and bc.py - the code for training and testing behavorial cloning.

For a quick start on testing our best performing model in the treechop environment, make sure to first follow in the installation instructions in the mineRL repo and install our requirements.txt.

Then you can run the following within the src file to run our best performing model in the TreeChop environement.

```bash test.sh```

All of our navigation code and baseline experiments can be found in the experiments/ subfolder within src/.

## Installation Guide
First make sure to have all of the needed requirements using our requirements.txt file.
```pip install -r requirements.txt```

You will likely encounter an issue with the installation of MineRL if you are on MacOS or windows. In order to circumvent this follow these instructions:

0. First follow install instructions on the official MineRL docs website
https://minerl.readthedocs.io/en/v0.4.4/tutorials/index.html

1. Install the follwing MixinGradle file that is missing from the repo's cache in order for Malmo to work:
https://drive.google.com/file/d/1z9i21_GQrewE0zIgrpHY5kKMZ5IzDt6U/view?usp=drive_link

2. Clone the mineRL repo locally and checkout their v0.4 branch that we are using.
```git clone https://github.com/minerllabs/minerl.git```
```cd minerl```
```git checkout v0.4```

3. Go into the build.gradle file
```cd minerl/Malmo/Minecraft```

4. Then update the build.gradle in the following way:

4.1 Add the following to the repositories section:
```
maven {
    url 'your path to the parent directory of the mixingradle file you installed'
}
```

4.2 Make sure your dependencies section looks like this:
```
dependencies {
        classpath 'org.ow2.asm:asm:6.0'
          classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61'){ // 0.6
            // Because forgegradle requires 6.0 (not -debug-all) while mixingradle depends on 5.0
            // and putting mixin right here will place it before forge in the class loader
            exclude group: 'org.ow2.asm', module: 'asm-debug-all'
        }

        classpath 'com.github.brandonhoughton:ForgeGradle:FG_2.2_patched-SNAPSHOT'
    }
```

5. Now you should be able to pip install using this directory and have no issues.


For training and testing various different models - use the test.sh file in order to run the appropriate model that you are interested in using. The file has examples for each of the scripts that you can run for the various purposes you may have:
```bash test.sh```


