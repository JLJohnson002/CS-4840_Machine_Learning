#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <iostream>

// int getinputi(int inputi1)
// {
//     return (int)inputi1;
// }

void stackOverflow(char arr[5]) {
    char newArr[100000]; // Large local array to consume stack space
    // stackOverflow(newArr); // Recursive call without base case
}

// char getinputc(char *inputc1)
// {
//     char inputc = inputc1[5];
//     return inputc;
// }gcc

// char getinputs(char inputs1)
// {
//     char inputs = (char)malloc((strlen(inputs1) - 5)sizeof(inputs1)); // i got desperate in this line
//     inputs = inputs1;
//     return inputs;
// }

int main()
{

    char arr[5] = "ice"; // Initial large array
    printf("Output for blahinteger: %s int \n", arr);
    char arr2[5] = "icicle"; // Initial large array
    printf("Output for integer: %s int \n", arr2);

    stackOverflow(arr);
    return 0;
    // integer value pushed to overflow - integer
    // printf("Output for integer: %d int \n", getinputi(21474836472));
    // printf("Output for integer: %d int \n", getinputi(2147483647));

    // character value pushed to overflow - stack
    // printf("Output for character: %s stack \n", getinputc("Ice"));
    // char inputc = getinputc("Icicle");
    // printf("Output for character: %s stack \n", inputc);

    // // heap overflow
    // char inputs = getinputs("lots of words");
    // printf("Output for characters: %s heap \n", inputs);

    // return 0;
}