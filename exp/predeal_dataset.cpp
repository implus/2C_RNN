/*************************************************************************
	> File Name: predeal.cpp
	> Author:   implus
	> Mail:     implusdream@gmail.com
	> Created Time: Tue 08 Mar 2016 06:46:53 PM PST
 ************************************************************************/

#include<iostream>
#include<bits/stdc++.h>
#include<tr1/unordered_map>
using namespace std;
using namespace std::tr1;

const int MAX_LEN = 2222222;
char buf[MAX_LEN], sentence[MAX_LEN];
unordered_map<string, int> word2idx, word2cnt;
unordered_map<int, string> idx2word;
unordered_map<int, int>    idx2cnt;
typedef pair<int, string> pis;


const string bos = "<S>";
const string eos = "</S>"; bool addEnd = false;
const string unk = "<unk>";

vector<vector<string> > trainvvs;

// 1: xxxx/train
// 2: xxxx/valid 
// 3: xxxx/test
// 4: xxxx/             -- output place
//
// out: xxxx/train_0.txt
// out: xxxx/valid_0.txt
// out: xxxx/test_0.txt
// out: xxxx/idx2cnt.txt xxxx/idx2word.txt
// out: xxxx/sortmapxy.txt
void prt(string info){
    cerr<<"----------------------------------- "<< info <<" ----------------------------------------------------"<<endl;
}
int main(int argc, char* argv[]){
    if(argc < 5) {
        cout<<"./predeal_dataset ../../dataset/big-data/es/train.txt ../../dataset/big-data/es/valid.txt ../../dataset/big-data/es/test.txt ../../dataset/big-data/es/"<<endl;
        return 1;
    }
    if(argc == 6 && string(argv[5]) == "addEnd") {
        cerr<<"Please check !!!!!!!!!!!!!!!!!!!!!!!  now need to add symbol <eos> for the end of each sentence !!!!!!!!!! Please make sure you want that !!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
        addEnd = true;
    }
    ios::sync_with_stdio(false);
    // 00. index , word2cnt, to erase some words
    prt("get word2cnt");
    trainvvs.push_back(vector<string>());
    sprintf(buf, "%s", argv[1]);
    cout<<buf<<" dealing..."<<endl;
    fstream fs(buf);
    int tot = 0;
    while(fs.getline(sentence, MAX_LEN)){
        string se = sentence;
        trainvvs[trainvvs.size() - 1].push_back(se);
        stringstream ss(se);
        string str;
        while(ss >> str){
            word2cnt[str]++;
        }
        if(addEnd) word2cnt[eos]++;
    }



    prt("fill vocabulary!");
    int idx = 0;
    // word2idx[bos] = ++idx; word2idx[eos] = ++idx; word2idx[unk] = ++idx; 
    // word2idx[unk] = ++idx; 
    // word2idx[eos] = ++idx;
    for(unordered_map<string, int>::iterator it = word2cnt.begin(); it != word2cnt.end(); it++){
        // if(it->second < 3) continue; // not limit to delete word by frequency
        // if(it->second == 1) continue; // occur 1 time is unk
        word2idx[it->first] = ++idx;
    }
    cout<<"!!!!!!!!!!!!vocabulary is "<<word2idx.size()<<endl;


    // 01. get train_xxx.txt
    prt("get train_xxx.txt");
    for (int f = 0; f < trainvvs.size(); f++){
        cerr<<"till now working on training file "<<f<<endl;
        sprintf(buf, "%s/train_%01d.txt", argv[4], f);
        fstream trainfs(buf, ios::out);
        vector<string> &trainvs = trainvvs[f];
        for (int i = 0; i < trainvs.size(); i++){
            if(i % 100000 == 0) cout<<i<<"finished, total sentence length = "<<trainvs.size()<<endl;
            // begin of sentence trainfs << word2idx[bos];
            string str = trainvs[i], word;
            stringstream ss(str);
            while(ss >> word){
                trainfs << " ";
                if(word2idx.find(word) == word2idx.end()) trainfs << word2idx[unk];
                else trainfs << word2idx[word];
                tot++;
            }
            if(addEnd) trainfs << " " << word2idx[eos] << "\n", tot++;
            else trainfs << "\n";
            //trainfs << " " << word2idx[eos] << "\n"; tot++;
        }
    }
    
    // 02. get idx2word, idx2cnt
    prt("generate idx2word.txt, idx2cnt.txt");
    sprintf(buf, "%s/idx2word.txt", argv[4]);
    fstream i2w(buf, ios::out);
    sprintf(buf, "%s/idx2cnt.txt", argv[4]);
    fstream i2c(buf, ios::out);
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


    // 03. get test_0.txt
    prt("generate text_0.txt");
    sprintf(buf, "%s/test_0.txt", argv[4]);
    fstream testfs(buf, ios::out);
    for(int i = 0; i < 1; i++){
        //sprintf(buf, "../../dataset/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en.heldout-%05d-of-00050", i);
        sprintf(buf, "%s", argv[3]);
        cout<<buf<<" dealing"<<endl;
        fstream fs(buf);
        while(fs.getline(sentence, MAX_LEN)){
            // testfs << word2idx[bos];
            string str = sentence, word;
            stringstream ss(str);
            while(ss >> word){
                testfs << " ";
                if(word2idx.find(word) == word2idx.end()) testfs << word2idx[unk];
                else testfs << word2idx[word];
            }
            if(addEnd) 
            testfs << " " << word2idx[eos] << "\n";
            else
            testfs << "\n";
            //testfs << " " << word2idx[eos] << "\n";
        }
    }

    // 04. get valid_0.txt
    prt("generate valid_0.txt");
    sprintf(buf, "%s/valid_0.txt", argv[4]);
    fstream validfs(buf, ios::out);
    for(int i = 1; i < 2; i++){
        sprintf(buf, "%s", argv[2]);
        cout<<buf<<" dealing"<<endl;
        fstream fs(buf);
        while(fs.getline(sentence, MAX_LEN)){
            // validfs << word2idx[bos];
            string str = sentence, word;
            stringstream ss(str);
            while(ss >> word){
                validfs << " ";
                if(word2idx.find(word) == word2idx.end()) validfs << word2idx[unk];
                else validfs << word2idx[word];
            }
            if(addEnd) 
            validfs << " " << word2idx[eos] << "\n";
            else
            validfs << "\n";
            //validfs << " " << word2idx[eos] << "\n";
        }
    }
    

    vector<pis> vp;
    int cnt_up3 = 0;
    for(unordered_map<string, int>::iterator it = word2idx.begin(); it != word2idx.end(); it++){
        idx2cnt[it->second] = word2cnt[it->first];
        vp.push_back(pis(word2cnt[it->first], it->first));
    }
    sort(vp.begin(), vp.end());
    reverse(vp.begin(), vp.end());
    
    int sum = 0;
    for(int i = 0; i < 200; i++){
        cout<<vp[i].second<<"\t"<<vp[i].first<<"|\t";
        sum += vp[i].first;
    }
    cout<<"\nSummary: big frequency 200 word sum = ===== = "<<sum<<endl<<" tot number of words in training set = "<<tot<<endl;
    

    prt("assign sortmapxy.txt");
    vector<string> vs;
    vector<vector<string> > vvs;
    for(unordered_map<string, int>::iterator it = word2idx.begin(); it != word2idx.end(); it++){
        vs.push_back(it->first);
    }
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
    sprintf(buf, "%s/sortmapxy.txt", argv[4]);
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
