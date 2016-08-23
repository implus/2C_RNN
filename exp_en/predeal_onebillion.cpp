/*************************************************************************
	> File Name: predeal.cpp
	> Author: 
	> Mail: 
	> Created Time: Tue 08 Mar 2016 06:46:53 PM PST
 ************************************************************************/

#include<iostream>
#include<bits/stdc++.h>
#include<tr1/unordered_map>
using namespace std;
using namespace std::tr1;

const int MAX_LEN = 222222;
char buf[MAX_LEN], sentence[MAX_LEN];
unordered_map<string, int> word2idx, word2cnt;
unordered_map<int, string> idx2word;
unordered_map<int, int>    idx2cnt;
typedef pair<int, string> pis;


const string bos = "<S>";
const string eos = "</S>";
const string unk = "<unk>";
// note that we must make vocabulary == 793471, let cnt>=3 

vector<vector<string> > trainvvs;
const int each = 99;

int main(){
    ios::sync_with_stdio(false);
    // 00. index , word2cnt, to erase some words
    cout<<"get word2cnt"<<endl;
    trainvvs.push_back(vector<string>());
    for(int i = 1; i < 100; i++){
        sprintf(buf, "../../dataset/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-%05d-of-00100", i);
        cout<<buf<<" going..."<<endl;
        fstream fs(buf);
        while(fs.getline(sentence, MAX_LEN)){
            string se = sentence;
            trainvvs[trainvvs.size() - 1].push_back(se);
            stringstream ss(se);
            string str;
            while(ss >> str){
                word2cnt[str]++;
            }
        }

        if(i % each == 0 && i != 99) {
            trainvvs.push_back(vector<string>());
        }
    }

    cout<<"fill vocabulary!~"<<endl;
    int idx = 0, tot = 0;
    word2idx[bos] = ++idx;
    word2idx[eos] = ++idx;
    word2idx[unk] = ++idx;
    for(unordered_map<string, int>::iterator it = word2cnt.begin(); it != word2cnt.end(); it++){
        if(it->second < 3) continue;
        word2idx[it->first] = ++idx;
    }
    cout<<"!!!!!!!!!!!!vocabulary is "<<word2idx.size()<<endl;


    // 01. get train_xxx.txt
    // system("mkdir -p ../../dataset/1-billion-word-language-modeling-benchmark-r13output/split99/");
    assert(trainvvs.size() == 1);
    for (int f = 0; f < trainvvs.size(); f++){
        cerr<<"till now working on training file "<<f<<endl;
        // sprintf(buf, "../../dataset/1-billion-word-language-modeling-benchmark-r13output/split99/train_%02d.txt", f);
        sprintf(buf, "../../dataset/1-billion-word-language-modeling-benchmark-r13output/train_%01d.txt", f);
        fstream trainfs(buf, ios::out);
        vector<string> &trainvs = trainvvs[f];
        for (int i = 0; i < trainvs.size(); i++){
            if(i % 100000 == 0) cout<<i<<"finished, tot = "<<trainvs.size()<<endl;
            // begin of sentence 
            trainfs << word2idx[bos];
            string str = trainvs[i], word;
            stringstream ss(str);
            while(ss >> word){
                trainfs << " ";
                if(word2idx.find(word) == word2idx.end()) trainfs << word2idx[unk];
                else trainfs << word2idx[word];
            }
            trainfs << " " << word2idx[eos] << "\n";
        }
    }
    
    // 02. get idx2word, idx2cnt
    cout<<" generate idx2word txt, idx2cnt txt"<<endl;
    fstream i2w("../../dataset/1-billion-word-language-modeling-benchmark-r13output/idx2word.txt", ios::out);
    fstream i2c("../../dataset/1-billion-word-language-modeling-benchmark-r13output/idx2cnt.txt", ios::out);
    for(unordered_map<string, int>::iterator it = word2idx.begin(); it != word2idx.end(); it++){
        idx2word[it->second] = it->first;
    }
    for (int i = 1; i <= idx; i++){
        i2w << idx2word[i] << "\n";
    }
    for (int i = 1; i <= idx; i++){
        i2c << word2cnt[idx2word[i]] << "\n";
    }
    assert(idx == (int)word2idx.size());


    // 03. get test.txt
    cout<<"generate test.txt"<<endl;
    fstream testfs("../../dataset/1-billion-word-language-modeling-benchmark-r13output/test_0.txt", ios::out);
    for(int i = 0; i < 1; i++){
        sprintf(buf, "../../dataset/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en.heldout-%05d-of-00050", i);
        cout<<buf<<" going"<<endl;
        fstream fs(buf);
        while(fs.getline(sentence, MAX_LEN)){
            testfs << word2idx[bos];
            string str = sentence, word;
            stringstream ss(str);
            while(ss >> word){
                testfs << " ";
                if(word2idx.find(word) == word2idx.end()) testfs << word2idx[unk];
                else testfs << word2idx[word];
            }
            testfs << " " << word2idx[eos] << "\n";
        }
    }

    // 04. get valid.txt
    cout<<"generate valid.txt"<<endl;
    fstream validfs("../../dataset/1-billion-word-language-modeling-benchmark-r13output/valid_0.txt", ios::out);
    for(int i = 1; i < 50; i++){
        sprintf(buf, "../../dataset/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en.heldout-%05d-of-00050", i);
        cout<<buf<<" going"<<endl;
        fstream fs(buf);
        while(fs.getline(sentence, MAX_LEN)){
            validfs << word2idx[bos];
            string str = sentence, word;
            stringstream ss(str);
            while(ss >> word){
                validfs << " ";
                if(word2idx.find(word) == word2idx.end()) validfs << word2idx[unk];
                else validfs << word2idx[word];
            }
            validfs << " " << word2idx[eos] << "\n";
        }
    }
    

    vector<pis> vp;
    int cnt_up3 = 0;
    for(unordered_map<string, int>::iterator it = word2idx.begin(); it != word2idx.end(); it++){
        idx2cnt[it->second] = word2cnt[it->first];
        vp.push_back(pis(word2cnt[it->first], it->first));
        if(word2cnt[it->first] >= 3) cnt_up3++;
    }
    cout<<"!!!!!!!!!!!!!!!!!!!cnt up3 = "<<cnt_up3<<endl;
    sort(vp.begin(), vp.end());
    reverse(vp.begin(), vp.end());
    
    int sum = 0;
    for(int i = 0; i < 800; i++){
        cout<<vp[i].second<<"\t"<<vp[i].first<<";\t";
        sum += vp[i].first;
    }
    cout<<"sum = ===== = "<<sum<<endl<<" tot number = "<<tot<<endl;
    
    vector<string> vs;
    vector<vector<string> > vvs;
    for(unordered_map<string, int>::iterator it = word2idx.begin(); it != word2idx.end(); it++){
        vs.push_back(it->first);
    }

    cerr<<"gao sortmapxy.txt!"<<endl;
    cerr<<"vocabulary size = "<<vs.size()<<endl;
    int base = ceil(sqrt(1.0 * vs.size()));
    cerr<<"base = "<<base<<endl;

    int id = 0;
    sort(vs.begin(), vs.end());

    for(int i = 0; i < base; i++){
        vvs.push_back(vector<string>());
        for(int j = 0; j < base; j++){
            if(id >= vs.size()) break;
            vvs[i].push_back(vs[id++]);
        }
    }

    for(int i = 0; i < base; i++){
        vector<string>& lx = vvs[i]; 
        for(int j = 0; j < lx.size(); j++){
            reverse(lx[j].begin(), lx[j].end());
        }
        sort(lx.begin(), lx.end());
    }

    // mapping
    sprintf(buf, "../../dataset/1-billion-word-language-modeling-benchmark-r13output/sortmapxy.txt");
    fstream fmap(buf, ios::out);
    for(int i = 0; i < base; i++){
        for(int j = 0; j < base; j++){
            if(j >= vvs[i].size()) fmap << "0 ";
            else {
                string t = vvs[i][j];
                reverse(t.begin(), t.end());
                fmap << word2idx[t] << " ";
            }
        }
        fmap << "\n";
    }
    assert(idx == (int)word2idx.size());
    
    return 0;
}
