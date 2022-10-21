#include <iostream>
#include <vector>

int main() {

	int N, enter;
	std::cin >> N;
	int max = 0, need = 0;

	for (int i = 0; i < N; i++) {
		
		std::cin >> enter;

		enter = (enter - 1) / 5;

		if (enter == 0) need -= 1;
		else need += enter;

		if (need > max) max = need; 
	}
	std::cout << max << std::endl;
}