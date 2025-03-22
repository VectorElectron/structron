#include <iostream>
#include <queue>
#include <random>
#include <chrono>

using namespace std;
using namespace std::chrono;

void heapTest(int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    priority_queue<double, vector<double>, greater<double>> pq;

    auto start = high_resolution_clock::now();


    for (int i = 0; i < n; ++i) {
        pq.push(dis(gen));
    }

    for (int i = 0; i < n; ++i) {
        pq.pop();
    }

    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Time taken: " << duration.count() << " milliseconds" << endl;
}

int main() {
    int n = 1000000;
    heapTest(n);
    return 0;
}