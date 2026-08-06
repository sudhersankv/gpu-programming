// CPU Implementation of Reduction
#include <stdio.h>

int main(){


int arr[5] = {1, 2, 3, 4, 5};

int n = sizeof(arr) / sizeof(arr[0]); // size of the array

int sum = 0;

for ( int i = 0; i < n; i++)
{
    sum += arr[i];
}

printf("Sum: %d\n", sum);
}
