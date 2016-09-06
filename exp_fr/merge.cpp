/*************************************************************************
	> File Name: merge.cpp
	> Author: 
	> Mail: 
	> Created Time: Mon 13 Jun 2016 03:35:42 PM CST
 ************************************************************************/

#include<iostream>
#include<bits/stdc++.h>
using namespace std;

typedef vector<vector<double> > VVD;
const int MAX_LEN = 12345678;
char sentence[MAX_LEN];
VVD gao(string file){
    fstream fs(file.c_str());
    VVD res;
    while(fs.getline(sentence, MAX_LEN)){
        stringstream ss(sentence);
        res.push_back(vector<double>());
        int w = res.size() - 1;
        double v;
        while(ss >> v){
            res[w].push_back(v);
        }
    }
    return res;
}

void add(VVD& res, VVD& pad){
    for(int i = 0; i < res.size(); i++)
    for(int j = 0; j < res[i].size(); j++){
        res[i][j] += pad[i][j];
    }
}

void out(VVD& a){
    cerr<<"wx = "<<a.size()<<" wy = "<<a[0].size()<<endl;
}

int main(int argc, char* argv[]){
    ios_base::sync_with_stdio(false);
    string des(argv[1]);
    cerr<<"there are "<<argc - 2<<" files to be merged."<<endl;
    // i == 2
    VVD res = gao(argv[2]);
    out(res);
    for(int i = 3; i < argc; i++){
        VVD pad = gao(argv[i]);
        out(pad);
        add(res, pad);
    }
    cerr<<"write res into file "<<des<<endl;
    
    fstream fw(des.c_str(), ios::out);
    for(int i = 0; i < res.size(); i++){
        for(int j = 0; j < res[i].size(); j++){
            fw << ' ' << res[i][j];
        }
        fw << '\n';
    }
    return 0;
}
