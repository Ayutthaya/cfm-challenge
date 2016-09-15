#include <cstdio>
#include <cstring>
#include <vector>

using namespace std;

int main()
{
    const char * filename = "/home/nath/Projects/Kaggle/data/uncompressed-data/train_numeric.csv";
    FILE * file;

    file = fopen(filename, "r");
    if (file==nullptr) {
        printf("Could not open file.\n");
        return 1;
    }

    const int buffersize = 1000000;
    char buffer [buffersize];

    char format [32];
    sprintf(format, "%c%ds", '%', buffersize-1);

    int n_cols = 0;

    // process first line
    if (fscanf(file, format, buffer) != -1) {
        int n_chars = strlen(buffer);
        for (int i=0; i<n_chars; i++)
            if (buffer[i]==',')
                n_cols++;
        printf("Number of columns: %d\n", n_cols);
    }
    else {
        fclose(file);
        printf("A problem occurred when reading the file.\n");
        return 1;
    }

    int value_count = 0;
    int count = 0;
    int res;
    do {
        res = fscanf(file, format, buffer);
        
        if (res == -1) {
            break;
        }

        count++;
        if (count%10000==0) {
            printf("%d lines read.\n", count);
        }

        int n_chars = strlen(buffer);
        int left=0, right=0;

        while (right < n_chars) {

            if (buffer[right]==',') {

                if (right-left > 1) {
                    value_count++;
                }

                left = right++;
            }

            else
                right++;
        }

    } while(res != -1);

    printf("Number of lines: %d\n", count);
    printf("Number of values: %d\n", value_count);

    printf("End of file.\n");
    fclose(file);

    return 0;
}
