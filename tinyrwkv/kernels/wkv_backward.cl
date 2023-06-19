__kernel void wkv_backward(global float* restrict const ret, constant const int* restrict const shape, constant const float* restrict const time_first, constant const float* restrict const time_decay, constant const float* restrict const key, constant const float* restrict const value, constant const float* restrict const wkv, constant const float* restrict const grad) {
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
    constant const float* restrict const y = wkv + offset;
    constant const float* restrict const gy = grad + offset;

    global float* restrict const gkp = ret + B * T * C * 2;
    global float* restrict const gvp = ret + B * T * C * 2 + 1;

    global float* restrict const gk = gkp + offset;
    global float* restrict const gv = gvp + offset;

    float q[1024], r[1024];

    // state
    float rgw = 0, rgu = 0, aa = 0, ga = 0, bb = 0, gb = 0, pp = -1e38f;

    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = k[ii];
        const float vv = v[ii];
        const float yy = y[ii];

        const float ww1 = u + kk;
        const float p1 = max(pp, ww1);
        const float e11 = exp(pp - p1);
        const float e21 = exp(ww1 - p1);
        const float qq = gy[ii] / fma(e11, bb, e21);
        rgw += (ga - gb * yy) * e11 * qq;
        rgu += (vv - yy) * e21 * qq;
        q[i] = qq;
        r[i] = ww1 - p1;

        const float ww2 = w + pp;
        const float p2 = max(ww2, kk);
        const float e12 = exp(ww2 - p2);
        const float e22 = exp(kk - p2);
        ga = fma(e12, aa, ga);
        aa = fma(e12, aa, e22 * vv);
        gb = fma(e12, bb, gb);
        bb = fma(e12, bb, e22);
        pp = p2;
    }

    global float* restrict const gu = ret + B * T * C * 0;
    global float* restrict const gw = ret + B * T * C * 1;

    const int offsetBC = b * C + c;
    gw[offsetBC] = rgw * w;
    gu[offsetBC] = rgu;

    // state
    aa = 0, bb = 0, pp = -1e38f;
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = k[ii];
        const float vv = v[ii];
        const float yy = y[ii];
        const float qq = q[i];
        const float rr = r[i];

        const float e11 = qq * exp(rr);
        const float e21 = exp(kk + pp);
        gk[ii] = e11 * (vv - yy) + e21 * (aa * vv + bb);
        gv[ii] = e11 + e21 * aa;

        const float ww = w + pp;
        const float www = rr - u - kk;
        const float p = max(ww, www);
        const float e12 = exp(ww - p);
        const float e22 = qq * exp(www - p);
        aa = fma(e12, aa, e22);
        bb = fma(e12, bb, e22 * yy);
        pp = p;
    }
}
