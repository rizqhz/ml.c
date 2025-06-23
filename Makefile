proof: include/activation.c include/loss.c include/regression.c tests/linear_regression.c
	@ clear; clang -O3 -I./. ${^} -o main.exe; ./main.exe; rm main.exe