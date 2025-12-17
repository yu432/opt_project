# opt_project

# WideResNet baseline
python main.py --model WRN --optimizer SGD

# ResNet baseline
python main.py --model ResNet50 --optimizer SGD

# WideResNet SAM
python main.py --model WRN --optimizer SAM --rho 0.3

# ResNet SAM
python main.py --model ResNet50 --optimizer SAM --rho 0.3

# WideResNet MSAM
python main.py --model WRN --optimizer MSAM --rho 3

# ResNet MSAM
python main.py --model ResNet50 --optimizer MSAM --rho 3

# WideResNet RAND_MSAM
python main.py --model WRN --optimizer RAND_MSAM --rho 3

# ResNet RAND_MSAM
python main.py --model ResNet50 --optimizer RAND_MSAM --rho 3
