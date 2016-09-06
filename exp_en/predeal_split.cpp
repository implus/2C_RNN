/*************************************************************************
	> File Name: predeal_split.cpp
	> Author: 
	> Mail: 
	> Created Time: Tue 07 Jun 2016 04:16:48 PM CST
 ************************************************************************/

#include<bits/stdc++.h>
#include<iostream>
using namespace std;

const int MAX_LEN = 1234567;
char sentence[MAX_LEN];

void gao(char * prefix, string file, int tot){
    string readin = string(prefix) + "/" + file + "_0.txt";
    cerr<<"readin "<<readin<<endl;
    fstream fs(readin.c_str());
    vector<string> vs;
    while(fs.getline(sentence, MAX_LEN)){
        vs.push_back(sentence);
    }
    int len = vs.size();
    cerr<<"total sentences number = "<<len<<endl;
    int one_part = ceil(1.0 * len / tot);
    int st = 0;
    for(int i = 1; i <= tot; i++){
        sprintf(sentence, "%s/%s_%d.txt", prefix, file.c_str(), i);
        fstream fo(sentence, ios::out);
        int last = min(st + one_part, len);
        for(int j = st; j < last; j++){
            fo << vs[j] << '\n';
        }
        st = last;
    }
}

int main(int argc, char* argv[]){
    // argv[1]/train_0.txt -> train_1.txt, train_2.txt ...
    //         valid
    //         test
    if(argc < 3) {
        cerr<<"Usage: ./predeal_split xx/xx/data/ 4"<<endl;
        return 0;
    }
    stringstream ss(argv[2]);
    int tot = 0;
    ss >> tot;
    cerr<<"split into "<<tot<<" files"<<endl;

    gao(argv[1], "train", tot);
    gao(argv[1], "valid", tot);
    gao(argv[1], "test",  tot);

    return 0;
}
