proof-linear: include/activation.c include/loss.c include/regression.c tests/linear_regression.c
	@ clear; clang -O3 -I./. ${^} -o main.exe; ./main.exe; rm main.exe

proof-logistic: include/activation.c include/loss.c include/regression.c tests/logistic_regression.c
	@ clear; clang -O3 -I./. ${^} -o main.exe; ./main.exe; rm main.exe