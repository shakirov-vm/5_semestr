#include <vector>
#include <iostream>

int main() {

	int N, Q, L, R, X;
	std::cin >> N >> Q;
	std::vector<size_t> result(N);

	for(int i = 0; i < N; i++) std::cin >> result[i];
	for(int i = 0; i < Q; i++) {
		std::cin >> L >> R >> X;
		for(int j = L; j <= R; j++) result[j] += X;
	}

	for(auto it : result) std::cout << it << " ";
	std::cout << std::endl;	
}