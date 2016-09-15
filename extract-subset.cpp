#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

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

    vector<string> column_names;
    int n_cols;

    // process first line
    if (fscanf(file, format, buffer) != -1) {
        int n_chars = strlen(buffer);
        int left = 0, right = 0;
        for (int right=0; right<=n_chars; right++) {
            if (right==n_chars || buffer[right]==',') {
                column_names.emplace_back(buffer+left, right-left);
                left=right;
            }
        }
        n_cols = column_names.size();
        printf("Number of columns: %d\n", n_cols);
    }
    else {
        fclose(file);
        printf("A problem occurred when reading the file.\n");
        return 1;
    }


    double threshold = 0.9;
    vector<int> similar_lines;
    vector<bool> first_line_indices(n_cols, false);
    int line_count = 0;
    int res;
    do {
        res = fscanf(file, format, buffer);
        
        if (res == -1) {
            break;
        }

        vector<bool> current_line_indices(n_cols, false);
        int col_count=0;
        int n_chars = strlen(buffer);
        int left=0, right=0;
        while (right < n_chars) {


            if (buffer[right]==',') {

                if (line_count==0) {
                    if (right-left > 1) {
                        first_line_indices[col_count]=true;
                    }
                }

                else {
                    if (right-left > 1) {
                        current_line_indices[col_count]=true;
                    }
                }

                col_count++;
                left = right;
            }
            right++;
        }

        int union_count = 0;
        int inter_count = 0;
        for (int i=0; i<n_cols; i++) {
            union_count += first_line_indices[i] || current_line_indices[i];
            inter_count += first_line_indices[i] && current_line_indices[i];
        }

        double jaccard_similarity = double(inter_count) / double(union_count);
        if (jaccard_similarity > threshold)
            similar_lines.push_back(line_count);

        line_count++;
        if (line_count%10000==0) {
            printf("%d lines read.\n", line_count);
            printf("%d lines similar to the first one.\n", similar_lines.size());
        }

    } while(res != -1);

    printf("Number of lines: %d\n", line_count);
    printf("Number of lines similar to the first line: %d\n", similar_lines.size());

    printf("End of file.\n");
    fclose(file);

    return 0;
}
