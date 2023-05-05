__kernel void wkv_forward(global float* restrict const ret, constant const int* restrict const shape, constant const float* restrict const time_first, constant const float* restrict const time_decay, constant const float* restrict const key, constant const float* restrict const value) {
    const int idx = get_global_id(0);
    const int B = shape[0];
    const int T = shape[1];
    const int C = shape[2];

    const int b = idx / C;
    const int c = idx % C;
    const int offset = b * T * C + c;

    const float u = time_first[c];
    const float w = time_decay[c];
    constant const float* restrict const k = key + offset;
    constant const float* restrict const v = value + offset;
    global float* restrict const wkv = ret + offset;

    // state
    float aa = 0, bb = 0, pp = -1e38;

    // do up till the last token in the loop
    for (int i = 0; i < T - 1; i++) {
        const int ii = i * C;
        const float kk = k[ii];
        const float vv = v[ii];

        const float ww1 = u + kk;
        const float p1 = max(pp, ww1);
        const float e11 = exp(pp - p1);
        const float e21 = exp(ww1 - p1);
        wkv[ii] = fma(e11, aa, e21 * vv) / fma(e11, bb, e21);

        const float ww2 = w + pp;
        const float p2 = max(ww2, kk);
        const float e12 = exp(ww2 - p2);
        const float e22 = exp(kk - p2);
        aa = fma(e12, aa, e22 * vv);
        bb = fma(e12, bb, e22);
        pp = p2;
    }

    // do the last token outside the loop to avoid having to compute state again
    const int ii = (T - 1) * C;
    const float kk = k[ii];
    const float vv = v[ii];

    const float ww1 = u + kk;
    const float p1 = max(pp, ww1);
    const float e11 = exp(pp - p1);
    const float e21 = exp(ww1 - p1);
    wkv[ii] = fma(e11, aa, e21 * vv) / fma(e11, bb, e21);
}
