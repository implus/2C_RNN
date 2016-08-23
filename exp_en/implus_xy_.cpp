/*************************************************************************
	> File Name: implus.cpp
	> Author:  implus for speed up
	> Mail: 
	> Created Time: Tue 29 Mar 2016 05:14:58 AM PDT
 ************************************************************************/

#include<iostream>
#include<bits/stdc++.h>
using namespace std;


vector<vector<double> > vvdx, vvdy;
struct node {
    int x, y;
    double val;
    node(){}
    node(int x, int y, double val):x(x), y(y), val(val){}
    bool operator<(const node& ths) const {
        return val < ths.val;
    }
};
vector<vector<node> > prob;
vector<vector<int> >  element_array;
struct qnode{
    int word, id, cnt;
    double sum, key;
    qnode(){}
    qnode(int word, int id, int cnt, double sum, double key):word(word), id(id), cnt(cnt), sum(sum), key(key){}
    bool operator<(const qnode& ths) const{
        return key < ths.key;
    }
};
priority_queue<qnode> pq;

const int MAX_LEN = 1234567;
char buf[MAX_LEN];

void file2vvd(char* filename, vector<vector<double> >& vvd){
    fstream f(filename, ios::in);
    while(f.getline(buf, MAX_LEN)){
        stringstream ss(buf); double val;
        vvd.push_back(vector<double>());
        while(ss >> val){
            vvd[vvd.size() - 1].push_back(val);
        }
    }
    f.close();
}
void element_array2file(char* filename){
    cerr<<filename<<" file to be saved"<<endl;
    fstream f(filename, ios::out);
    for(int i = 0; i < element_array.size(); i++){
        for(int j = 0; j < element_array[i].size(); j++){
            f << element_array[i][j] + 1 << " ";
        }
        f << "\n";
    }
    f.close();
}

vector<string> idx2word;
void idx2word2idx2word(){
    fstream f("../public/idx2word", ios::in);
    string str;
    while(f >> str){
        idx2word.push_back(str);
    }
}

//const int vocab_size = 10000;
//const int vocab_sqrt = 100;

int main(int argc, char* argv[]){
    // 1, 2, 3
    idx2word2idx2word();
    cerr<<" read files into vvdx vvdy"<<endl;
    file2vvd(argv[1], vvdx);
    file2vvd(argv[2], vvdy);

    int vocab_size = vvdx.size();
    int vocab_sqrt = vvdx[0].size();
    cerr<<"vocab_size = "<<vocab_size<<endl;
    cerr<<"vocab_sqrt = "<<vocab_sqrt<<endl;

    //generate prob
    cerr<<"generate prob"<<endl;
    for(int word = 0; word < vocab_size; word++){
        prob.push_back(vector<node>());
        for(int x = 0; x < vocab_sqrt; x++){
            for(int y = 0; y < vocab_sqrt; y++){
                node d(x, y, vvdx[word][x] + vvdy[word][y]);
                prob[word].push_back(d);
            }
        }
        sort(prob[word].begin(), prob[word].end());
        if(word % 1000 == 0) cerr<<" word = "<<word<<" finished!"<<endl;
    }

    cerr<<"prob size = "<<prob.size()<<","<<prob[0].size()<<endl;
    for(int i = 0; i < prob.size(); i++){
        vector<node>& vn = prob[i];
        double sum = 0;
        for(int j = 1; j < vn.size(); j++){
            sum += vn[j].val;
            assert(vn[j].val >= vn[j - 1].val);
        }
        qnode one(i, 0, vn.size() - 1, sum, 0);
        one.key = one.sum/one.cnt - prob[one.word][one.id].val;
        pq.push(one);
    }

    for(int i = 0; i < vocab_sqrt; i++){
        element_array.push_back(vector<int>(vocab_sqrt, -1));
    }

    int finished = 0;
    cerr<<"queue begin!"<< pq.size() <<endl;
    while( pq.size() > 0 ) {
        qnode one = pq.top(); pq.pop();
        node  pos = prob[one.word][one.id];
        if(element_array[pos.x][pos.y] > 0){
            one.cnt -= 1;
            one.sum -= pos.val;
            one.id  += 1;
            one.key = one.sum/one.cnt - prob[one.word][one.id].val;
            pq.push(one);
        }else{
            element_array[pos.x][pos.y] = one.word;
            if(one.id == 0){
                sprintf(buf, "%30s\t locate in (%5d, %5d) one.id = %5d; finished = %10d\n", idx2word[one.word].c_str(), pos.x, pos.y, one.id, ++finished);
                cerr<<buf;
            }
        }
    }
    element_array2file(argv[3]);

    return 0;
}
