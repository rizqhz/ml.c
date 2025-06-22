proof: include/regression.c include/loss.c tests/linear_regression.c
	@ clear; clang -O3 ${^} -I./. -I./include -o main.exe; ./main.exe; rm main.exe