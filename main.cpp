#include "main.h"

using namespace std;

typedef struct mat
{
    int row;
    int column;
    float** matrix;

    mat(int r, int c){
        float** tempmat = new float*[r];
        for(int i=0;i<r;i++){
            tempmat[i] = new float[c];
            for(int j=0;j<c;j++){
                tempmat[i][j]=0.0;
            }
        }
        row = r;
        column = c;
        matrix = tempmat;
        delete tempmat;
        tempmat = NULL;
    }

    mat(float** mats, int r, int c){
        matrix = mats;
        row = r;
        column = c;
    }

    mat(float* arr, int length){
        float** tempmat = new float*[1];
        tempmat[0] = new float[length];
        for(int i=0;i<length;i++){
            tempmat[0][i]=arr[i];
        }
        row = 1;
        column = length;
        matrix = tempmat;
        delete tempmat;
        tempmat = NULL;
    }

    void insertrow(float* arr){
        float** tempmat = new float*[row+1];
        for(int i=0;i<row;i++){
            tempmat[i] = new float[column];
            for(int j=0;j<column;j++){
                tempmat[i][j]=matrix[i][j];
            }
        }
        for(int j=0;j<column;j++){
            tempmat[row][j] = arr[j];
        }
        matrix = tempmat;
        row = row+1;
        delete tempmat;
        tempmat = NULL;
    }
    void eye(){
        for(int i=0;i<row;i++){
            matrix[i][i]=1;
        }
    }


    void transpose(){
        float** tempmat = new float*[column];
        for(int i=0;i<column;i++){
            tempmat[i] = new float[row];
            for(int j=0;j<row;j++){
                tempmat[i][j] = matrix[j][i];
            }
        }
        int temp = row;
        row = column;
        column = temp;
        matrix = tempmat;
        delete tempmat;
    }

    ~mat(){}
}mat, *matPointer;

mat matminus(mat x,mat y){
    mat minusm(x.row,x.column);
    for(int i=0;i<x.row;i++){
        for(int j=0;j<x.column;j++){
                minusm.matrix[i][j]=x.matrix[i][j]-y.matrix[i][j];
        }
    }
    return minusm;
}

mat multiple(mat x, mat y){
    mat multi(x.row,y.column);
    if(x.column!=y.row){
        return multi;
    }
    for(int i=0;i<x.row;i++){
        for(int j=0;j<y.column;j++){
            multi.matrix[i][j]=0;
            for(int k=0;k<x.column;k++){
                multi.matrix[i][j]+=x.matrix[i][k]*y.matrix[k][j];
            }
        }
    }
    return multi;
}

mat transpose(mat x){
    mat trans(x.column,x.row);
    for(int i=0;i<x.column;i++){
        for(int j=0;j<x.row;j++){
            trans.matrix[i][j] = x.matrix[j][i];
        }
    }
    return trans;
}

mat inverse(mat x){
    int dim = x.row;
    int n=dim;
    vector<vector<float> > a(n, vector<float>(n));
    vector<vector<float> > L(n, vector<float>(n));
    vector<vector<float> > U(n, vector<float>(n));
    vector<vector<float> > out1(n, vector<float>(n));
    vector<vector<float> > r(n, vector<float>(n));
    vector<vector<float> > u(n, vector<float>(n));

	int k,i,j;
	float s,t;

	for(i=0;i<n;i++){
        for (j = 0; j < n; j++){
            a[i][j] = x.matrix[i][j];
        }
    }

    for(j=0;j<n;j++)
        a[0][j]=a[0][j];  //����U����ĵ�һ��

    for(i=1;i<n;i++)
        a[i][0]=a[i][0]/a[0][0];   //����L����ĵ�1��
    for(k=1;k<n;k++)
    {
        for(j=k;j<n;j++)
        {
            s=0;
            for (i=0;i<k;i++)
                s=s+a[k][i]*a[i][j];   //�ۼ�
            a[k][j]=a[k][j]-s; //����U���������Ԫ��
        }
        for(i=k+1;i<n;i++)
        {
            t=0;
            for(j=0;j<k;j++)
                t=t+a[i][j]*a[j][k];   //�ۼ�
            a[i][k]=(a[i][k]-t)/a[k][k];    //����L���������Ԫ��
        }
    }
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
        {
            if(i>j)
            {
                L[i][j]=a[i][j];
                U[i][j]=0;
            }//���i>j��˵���д����У��������������ǲ��֣��ó�L��ֵ��U��//Ϊ0
            else
            {
                U[i][j]=a[i][j];
                if(i==j)
                    L[i][j]=1;  //�������i<j��˵����С���У��������������ǲ��֣��ó�U��//ֵ��L��Ϊ0
                else
                    L[i][j]=0;
            }
        }
        if(U[1][1]*U[2][2]*U[3][3]*U[4][4]==0)
        {
            printf("\n����󲻴���");
            return x;
        }

        /////////////////////��L��U�������
        for (i=0;i<n;i++) //�����U����u
        {
            u[i][i]=1/U[i][i];//�Խ�Ԫ�ص�ֵ��ֱ��ȡ����
            for (k=i-1;k>=0;k--)
            {
                s=0;
                for (j=k+1;j<=i;j++)
                    s=s+U[k][j]*u[j][i];
                u[k][i]=-s/U[k][k];//�������㣬���е������εõ�ÿһ��ֵ��
            }
        }
        for (i=0;i<n;i++) //�����L����r
        {
            r[i][i]=1; //�Խ�Ԫ�ص�ֵ��ֱ��ȡ����������Ϊ1
            for (k=i+1;k<n;k++)
            {
                for (j=i;j<=k-1;j++)
                    r[k][i]=r[k][i]-L[k][j]*r[j][i];   //�������㣬����˳�����εõ�ÿһ��ֵ
            }
        }


        for(i=0;i<n;i++)
        {
            for(j=0;j<n;j++)
            {out1[i][j]=0;}
        }
        for(i=0;i<n;i++)
        {
            for(j=0;j<n;j++)
            {
                for(k=0;k<n;k++)
                {
                    out1[i][j]+=u[i][k]*r[k][j];
                }
            }
        }

    mat out(n,n);
    for(i=0;i<n;i++){
        for (j = 0; j < n; j++){
            out.matrix[i][j] = out1[i][j];
        }
    }

    return out;

 }


int myadf(float* stock, int length){
    float t2[3] = {-1.95,-2.86,-3.41};
    float* diff = new float[length-1];
    for(int i=0;i<length-1;i++){
        diff[i] = stock[i+1] - stock[i];
    }

    int pmax = 12*(sqrt(sqrt(length/100)));
    int n = length-pmax-1;
    float* y = new float[n];
    float* yy = new float[n];
    float* x = new float[n];
    float* t = new float[n];
    for(int i=0;i<n;i++){
        y[i]=diff[pmax+i];
        yy[i] = stock[pmax+i];
        x[i]=1;
        t[i]=i;
    }

    mat Y(y,n);
    mat X(x,n);
    mat X3(X);
    X3.insertrow(t);
    X3.insertrow(yy);
    int bestp3 = pmax;
    for(int i=0;i<bestp3-1;i++){
        float* temp = new float[n];
        for(int j=length-2-n-i;j<length-2-i;j++){
            temp[j-length-2-n-i]=diff[j];
        }
        X3.insertrow(temp);
        delete temp;
        temp = NULL;
    }

    mat X3trans(transpose(X3).matrix,X3.column,X3.row);
    mat X3inv(multiple(X3,X3trans).matrix,X3.row,X3.row);
    mat B3(multiple(X3inv,multiple(X3,transpose(Y))).matrix,X3.row,1);
    mat E3(matminus(multiple(X3trans,B3),transpose(Y)).matrix,X3.row,1);
    float SSR3;
    SSR3 = 0.0;
    for(int i=0;i<n;i++){
        SSR3 += E3.matrix[i][0]*E3.matrix[i][0];
    }
    float t3gamma=0.0;
    t3gamma = B3.matrix[2][0] / ( sqrt( X3inv.matrix[2][2]*SSR3/(n-pmax-bestp3-2) ) );
    if (t3gamma<t2[2])
        return 1;
    else
        return 0;
}

//�˺�������ͬ���߳�ͬʱ���е���
void threadRightpairs(float pf[100][10000], int pfIndex[100], int length, int num, mypairsPointer pairs, int slices, int tvalues, int tvalues2,int core,int confinterval){
    //pairs �ļ���
    static int index = 0;
    static std::mutex mtx;

    int tempLength = num - 1;
    int start = slices*(tempLength/core);
    int len = (slices+1)*(tempLength/core);

    for(int m = start; m < len; m++){
        for(int n = m + 1; n < num; n++){
            //�ع����: ����б��beta
            float numerator = 0;
            float denominator = 0;
            float xsum = 0;
            float ysum = 0;
            for(int i = 0; i < length; i++){
                xsum = xsum + pf[m][i];
                ysum = ysum + pf[n][i];
            }
            float xmean = xsum/length;
            float ymean = ysum/length;
            for(int i = 0; i < length; i++){
                numerator = numerator + (pf[m][i]-xmean)*(pf[n][i] - ymean);
                denominator = denominator + (pf[m][i]-xmean)*(pf[m][i] - xmean);
            }
            float beta = numerator/denominator;

            //�ع����: ����ؾ�alpha
            float alpha = ymean - beta*xmean;

            //�ع����: ����в�epsilon
            float *epsilon = new float[length];
            for(int j = 0; j < length; j++){
                epsilon[j] = pf[n][j] - beta*pf[m][j] - alpha;
            }

            int test = myadf(epsilon,length);
            if(test==1){
                //����ӻ�����
                mtx.lock();
                pairs->total = index + 1;
                pairs->pindex1[index] = pfIndex[m];
                pairs->pindex2[index] = pfIndex[n];
                index++;
                mtx.unlock();
            }
            delete[] epsilon;

			}


        }
    }




//c++�ӿں���
DLL_EXPORT mypairsPointer rightpairs(float pf[100][10000], int pfIndex[100], int length,  int num, int tvalues, int tvalues2,int core,int confinterval){

    mypairsPointer pairs = (mypairsPointer)malloc(sizeof(mypairs));
    //coreΪcpu������   ����core = 4, num = 101 ����Ҫ�� ��ѭ��һ����100�� ��1~25��ѭ������һ���߳�, 26~50��ѭ�����ڶ����߳���, һ������
    //�����߳���
    std::vector<std::thread*> threads;
    for(int i = 0; i < core; i++){
        //, tvalues, tvalues2
        std::thread *t = new std::thread(threadRightpairs, pf, pfIndex, length, num, pairs, i, tvalues, tvalues2,core,confinterval);//�Ѳ����������������߳�
        threads.push_back(t);
    }
    for(auto it = threads.begin(); it != threads.end(); ++it){
        (*it)->join();//�ȴ����е��߳�ִ�����
    }


    return pairs;
}

DLL_EXPORT mypairsPointer slowpairs(float pf[100][10000], int pfIndex[100], int length, int num,int tvalues, int tvalues2, int confinterval){
    mypairsPointer pairs = (mypairsPointer)malloc(sizeof(mypairs));
    static int index = 0;

    int len = num - 1;

    for(int m = 0; m < len; m++){
        for(int n = m + 1; n < num; n++){
            //�ع����: ����б��beta
            float numerator = 0;
            float denominator = 0;
            float xsum = 0;
            float ysum = 0;
            for(int i = 0; i < length; i++){
                xsum = xsum + pf[m][i];
                ysum = ysum + pf[n][i];
            }
            float xmean = xsum/length;
            float ymean = ysum/length;
            for(int i = 0; i < length; i++){
                numerator = numerator + (pf[m][i]-xmean)*(pf[n][i] - ymean);
                denominator = denominator + (pf[m][i]-xmean)*(pf[m][i] - xmean);
            }
            float beta = numerator/denominator;

            //�ع����: ����ؾ�alpha
            float alpha = ymean - beta*xmean;

            //�ع����: ����в�epsilon
            float *epsilon = new float[length];
            for(int j = 0; j < length; j++){
                epsilon[j] = pf[n][j] - beta*pf[m][j] - alpha;
            }

            int test = myadf(epsilon,length);
            if(test==1){
                int pnum = index;
                pairs->total = pnum + 1;
                pairs->pindex1[index] = pfIndex[m];
                pairs->pindex2[index] = pfIndex[n];
                index++;
            }
            delete[] epsilon;

        }
    }

    return pairs;
}








