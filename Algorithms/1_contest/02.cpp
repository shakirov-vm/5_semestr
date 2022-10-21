#include <vector>
#include <iostream>
#include <algorithm>

void enter_vector(std::vector<int>& vec) {

	int enter;

	for(;;) {

		std::cin >> enter;
		if (enter == 0) break;
		vec.push_back(enter);
	}
}

int main() {

	std::vector<int> A;
	std::vector<int> B;
	std::vector<int> result;

	enter_vector(A);
	enter_vector(B);

	std::sort(A.begin(), A.end());
	std::sort(B.begin(), B.end());

	auto it_A = A.begin();
	auto it_B = B.begin();

	while(it_A != A.end() && it_B != B.end()) {

		if (*it_A == *it_B) {
			it_A++; it_B++;
		}

		else if (*it_A > *it_B) {
			result.push_back(*it_B);
			it_B++;
		}

		else {
			result.push_back(*it_A);
			it_A++;
		}
	}
	for(; it_A != A.end(); it_A++) result.push_back(*it_A);
	for(; it_B != B.end(); it_B++) result.push_back(*it_B);

	for(auto it : result) std::cout << it << " ";
	std::cout << std::endl;	
}