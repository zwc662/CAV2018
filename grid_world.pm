dtmc

const int states = 322;
const int actions = 3;

module grid_world

	s : [0..321];

	[] (s= 0) = true -> 1.0 : (s'=321);
	[] (s= 1) = true -> 1.0 : (s'=321);
	[] (s= 2) = true -> 1.0 : (s'=321);
	[] (s= 3) = true -> 1.0 : (s'=321);
	[] (s= 4) = true -> 1.0 : (s'=321);
	[] (s= 5) = true -> 1.0 : (s'=5);
	[] (s= 6) = true -> 1.0 : (s'=6);
	[] (s= 7) = true -> 1.0 : (s'=7);
	[] (s= 8) = true -> 1.0 : (s'=8);
	[] (s= 9) = true -> 1.0 : (s'=9);
	[] (s= 10) = true -> 1.0 : (s'=10);
	[] (s= 11) = true -> 1.0 : (s'=11);
	[] (s= 12) = true -> 1.0 : (s'=12);
	[] (s= 13) = true -> 1.0 : (s'=13);
	[] (s= 14) = true -> 1.0 : (s'=14);
	[] (s= 15) = true -> 1.0 : (s'=15);
	[] (s= 16) = true -> 1.0 : (s'=321);
	[] (s= 17) = true -> 1.0 : (s'=321);
	[] (s= 18) = true -> 1.0 : (s'=321);
	[] (s= 19) = true -> 1.0 : (s'=321);
	[] (s= 20) = true -> 1.0 : (s'=321);
	[] (s= 21) = true -> 0.103448275862 : (s'=21) + 0.413793103448 : (s'=22) + 0.48275862069 : (s'=24);
	[] (s= 22) = true -> 0.125 : (s'=22) + 0.153225806452 : (s'=23) + 0.721774193548 : (s'=24);
	[] (s= 23) = true -> 0.876712328767 : (s'=24) + 0.109589041096 : (s'=40) + 0.013698630137 : (s'=41);
	[] (s= 24) = true -> 0.458823529412 : (s'=24) + 0.517647058824 : (s'=25) + 0.0235294117647 : (s'=41);
	[] (s= 25) = true -> 0.15625 : (s'=26) + 0.65625 : (s'=42) + 0.1875 : (s'=43);
	[] (s= 26) = true -> 1.0 : (s'=43);
	[] (s= 27) = true -> 1.0 : (s'=27);
	[] (s= 28) = true -> 1.0 : (s'=28);
	[] (s= 29) = true -> 1.0 : (s'=29);
	[] (s= 30) = true -> 1.0 : (s'=30);
	[] (s= 31) = true -> 1.0 : (s'=31);
	[] (s= 32) = true -> 1.0 : (s'=321);
	[] (s= 33) = true -> 1.0 : (s'=321);
	[] (s= 34) = true -> 1.0 : (s'=321);
	[] (s= 35) = true -> 1.0 : (s'=321);
	[] (s= 36) = true -> 1.0 : (s'=321);
	[] (s= 37) = true -> 0.714285714286 : (s'=22) + 0.285714285714 : (s'=38);
	[] (s= 38) = true -> 0.0128900949796 : (s'=23) + 0.921981004071 : (s'=39) + 0.0651289009498 : (s'=40);
	[] (s= 39) = true -> 0.894736842105 : (s'=40) + 0.105263157895 : (s'=56);
	[] (s= 40) = true -> 1.39999123405E-13 : (s'=40) + 0.362905818065 : (s'=41) + 3.2144005143E-4 : (s'=42) + 0.636451301832 : (s'=57) + 3.2144005143E-4 : (s'=58);
	[] (s= 41) = true -> 0.0496894409938 : (s'=42) + 0.948535936114 : (s'=58) + 0.00177462289264 : (s'=59);
	[] (s= 42) = true -> 0.0487804878049 : (s'=43) + 0.926829268293 : (s'=59) + 0.0121951219512 : (s'=60) + 0.0121951219512 : (s'=75);
	[] (s= 43) = true -> 0.932203389831 : (s'=60) + 0.0677966101695 : (s'=76);
	[] (s= 44) = true -> 1.0 : (s'=44);
	[] (s= 45) = true -> 1.0 : (s'=45);
	[] (s= 46) = true -> 1.0 : (s'=46);
	[] (s= 47) = true -> 1.0 : (s'=47);
	[] (s= 48) = true -> 1.0 : (s'=321);
	[] (s= 49) = true -> 1.0 : (s'=321);
	[] (s= 50) = true -> 1.0 : (s'=321);
	[] (s= 51) = true -> 1.0 : (s'=321);
	[] (s= 52) = true -> 1.0 : (s'=321);
	[] (s= 53) = true -> 0.287733797303 : (s'=38) + 0.00587211831231 : (s'=39) + 0.680078294911 : (s'=54) + 0.0263157894737 : (s'=55);
	[] (s= 54) = true -> 0.0642046825972 : (s'=38) + 0.392469225199 : (s'=39) + 0.0941346850109 : (s'=54) + 0.449191407193 : (s'=55);
	[] (s= 55) = true -> 0.0284115368059 : (s'=40) + 8.60955660783E-4 : (s'=55) + 0.964700817908 : (s'=56) + 0.00344382264313 : (s'=57) + 0.00258286698235 : (s'=72);
	[] (s= 56) = true -> 0.590909090909 : (s'=57) + 0.409090909091 : (s'=73);
	[] (s= 57) = true -> 2.21933582623E-13 : (s'=57) + 0.52693320565 : (s'=58) + 4.78812544889E-4 : (s'=73) + 0.47210916926 : (s'=74) + 4.78812544889E-4 : (s'=75);
	[] (s= 58) = true -> 0.0202366127024 : (s'=59) + 0.00186799501868 : (s'=74) + 0.97198007472 : (s'=75) + 0.00591531755915 : (s'=91);
	[] (s= 59) = true -> 0.876923076923 : (s'=76) + 0.123076923077 : (s'=92);
	[] (s= 60) = true -> 1.0 : (s'=93);
	[] (s= 61) = true -> 1.0 : (s'=61);
	[] (s= 62) = true -> 1.0 : (s'=62);
	[] (s= 63) = true -> 1.0 : (s'=63);
	[] (s= 64) = true -> 1.0 : (s'=64);
	[] (s= 65) = true -> 0.8 : (s'=34) + 0.2 : (s'=50);
	[] (s= 66) = true -> 0.190476190476 : (s'=34) + 0.380952380952 : (s'=35) + 0.380952380952 : (s'=50) + 0.047619047619 : (s'=51) + 9.99866855977E-13 : (s'=66);
	[] (s= 67) = true -> 0.0833333333333 : (s'=36) + 0.0277777777778 : (s'=51) + 0.888888888889 : (s'=52);
	[] (s= 68) = true -> 0.0196454240537 : (s'=52) + 0.942740776234 : (s'=53) + 0.0376137997125 : (s'=69);
	[] (s= 69) = true -> 0.6 : (s'=53) + 0.2 : (s'=54) + 0.138461538462 : (s'=69) + 0.0615384615385 : (s'=70);
	[] (s= 70) = true -> 0.015625 : (s'=54) + 0.5234375 : (s'=55) + 0.015625 : (s'=70) + 0.4453125 : (s'=71);
	[] (s= 71) = true -> 0.21875 : (s'=55) + 0.46875 : (s'=71) + 0.3125 : (s'=72);
	[] (s= 72) = true -> 0.294683544304 : (s'=72) + 0.22835443038 : (s'=73) + 0.239493670886 : (s'=88) + 0.23746835443 : (s'=89);
	[] (s= 73) = true -> 0.124345549738 : (s'=73) + 0.176701570681 : (s'=74) + 0.13612565445 : (s'=89) + 0.562827225131 : (s'=90);
	[] (s= 74) = true -> 0.0131698455949 : (s'=74) + 0.288374205268 : (s'=75) + 0.063578564941 : (s'=90) + 0.633969118983 : (s'=91) + 9.08265213442E-4 : (s'=107);
	[] (s= 75) = true -> 0.101193565127 : (s'=91) + 0.573689673067 : (s'=92) + 0.0272444213804 : (s'=107) + 0.297872340426 : (s'=108);
	[] (s= 76) = true -> 3.9990233347E-13 : (s'=76) + 0.0222222222222 : (s'=92) + 0.444444444444 : (s'=93) + 0.0444444444444 : (s'=108) + 0.488888888889 : (s'=109);
	[] (s= 77) = true -> 1.0 : (s'=77);
	[] (s= 78) = true -> 1.0 : (s'=78);
	[] (s= 79) = true -> 1.0 : (s'=79);
	[] (s= 80) = true -> 1.0 : (s'=80);
	[] (s= 81) = true -> 1.0 : (s'=50);
	[] (s= 82) = true -> 0.346153846154 : (s'=50) + 0.0769230769231 : (s'=51) + 0.423076923077 : (s'=66) + 0.153846153846 : (s'=67);
	[] (s= 83) = true -> 0.142857142857 : (s'=52) + 0.0714285714286 : (s'=67) + 0.785714285714 : (s'=68) + 3.9990233347E-13 : (s'=83);
	[] (s= 84) = true -> 0.0337612933904 : (s'=52) + 0.770090347123 : (s'=68) + 0.183309557775 : (s'=69) + 0.0038040893961 : (s'=84) + 0.00903471231574 : (s'=85);
	[] (s= 85) = true -> 0.370408163265 : (s'=69) + 0.191836734694 : (s'=70) + 0.0969387755102 : (s'=85) + 0.340816326531 : (s'=86);
	[] (s= 86) = true -> 0.211267605634 : (s'=70) + 0.183098591549 : (s'=71) + 0.492957746479 : (s'=86) + 0.112676056338 : (s'=87);
	[] (s= 87) = true -> 0.654562383613 : (s'=71) + 0.0111731843575 : (s'=72) + 0.316573556797 : (s'=87) + 0.0176908752328 : (s'=88);
	[] (s= 88) = true -> 0.219390926041 : (s'=88) + 0.734617775016 : (s'=89) + 0.0136730888751 : (s'=104) + 0.0323182100684 : (s'=105);
	[] (s= 89) = true -> 0.5 : (s'=89) + 0.333333333333 : (s'=105) + 0.166666666667 : (s'=106);
	[] (s= 90) = true -> 0.2 : (s'=90) + 0.24 : (s'=106) + 0.56 : (s'=107);
	[] (s= 91) = true -> 0.17031500267 : (s'=107) + 0.635344367325 : (s'=108) + 0.00106780565937 : (s'=123) + 0.193272824346 : (s'=124);
	[] (s= 92) = true -> 1.99951166735E-13 : (s'=92) + 0.195901639344 : (s'=108) + 0.0684426229508 : (s'=109) + 0.470901639344 : (s'=124) + 0.264754098361 : (s'=125);
	[] (s= 93) = true -> 0.111111111111 : (s'=109) + 0.888888888889 : (s'=125);
	[] (s= 94) = true -> 1.0 : (s'=94);
	[] (s= 95) = true -> 1.0 : (s'=95);
	[] (s= 96) = true -> 1.0 : (s'=96);
	[] (s= 97) = true -> 0.5 : (s'=65) + 0.5 : (s'=66);
	[] (s= 98) = true -> 0.0671641791045 : (s'=66) + 0.410447761194 : (s'=67) + 0.0746268656716 : (s'=82) + 0.44776119403 : (s'=83);
	[] (s= 99) = true -> 0.0869565217391 : (s'=67) + 0.260869565217 : (s'=68) + 0.282608695652 : (s'=83) + 0.369565217391 : (s'=84) + 8.99946783761E-13 : (s'=99);
	[] (s= 100) = true -> 0.0240759579518 : (s'=68) + 3.39097999322E-4 : (s'=83) + 0.973889454052 : (s'=84) + 0.00169548999661 : (s'=85) + 2.9639769784E-13 : (s'=100);
	[] (s= 101) = true -> 0.333333333333 : (s'=85) + 0.333333333333 : (s'=86) + 1.16677038401E-12 : (s'=101) + 0.333333333333 : (s'=102);
	[] (s= 102) = true -> 0.469672397016 : (s'=86) + 6.48718780409E-4 : (s'=87) + 0.0470321115796 : (s'=101) + 0.482646772624 : (s'=102);
	[] (s= 103) = true -> 0.0350515463917 : (s'=87) + 1.18593565666E-13 : (s'=103) + 0.847422680412 : (s'=104) + 0.117525773196 : (s'=120);
	[] (s= 104) = true -> 0.475 : (s'=104) + 0.15 : (s'=105) + 0.35 : (s'=120) + 0.025 : (s'=121);
	[] (s= 105) = true -> 0.458149779736 : (s'=105) + 0.0396475770925 : (s'=106) + 0.36563876652 : (s'=121) + 0.136563876652 : (s'=122);
	[] (s= 106) = true -> 0.177453921989 : (s'=106) + 0.0107158165452 : (s'=107) + 0.0775825117874 : (s'=122) + 0.734247749679 : (s'=123);
	[] (s= 107) = true -> 0.521739130435 : (s'=123) + 0.347826086957 : (s'=124) + 0.0434782608696 : (s'=139) + 0.086956521739 : (s'=140);
	[] (s= 108) = true -> 0.595493934142 : (s'=124) + 0.169150779896 : (s'=125) + 0.0457538994801 : (s'=140) + 0.189601386482 : (s'=141);
	[] (s= 109) = true -> 0.0357142857143 : (s'=125) + 0.964285714286 : (s'=141);
	[] (s= 110) = true -> 1.0 : (s'=110);
	[] (s= 111) = true -> 1.0 : (s'=111);
	[] (s= 112) = true -> 1.0 : (s'=112);
	[] (s= 113) = true -> 1.0 : (s'=81);
	[] (s= 114) = true -> 0.333333333333 : (s'=82) + 0.222222222222 : (s'=83) + 0.222222222222 : (s'=98) + 0.222222222222 : (s'=99) + 9.9997787828E-13 : (s'=114);
	[] (s= 115) = true -> 0.212340188994 : (s'=83) + 0.00222345747638 : (s'=84) + 0.624235686492 : (s'=99) + 0.161200667037 : (s'=100) + 6.19948536951E-13 : (s'=115);
	[] (s= 116) = true -> 0.0212765957447 : (s'=83) + 0.0212765957447 : (s'=84) + 0.0711665443874 : (s'=99) + 0.884812912693 : (s'=100) + 0.00146735143067 : (s'=116);
	[] (s= 117) = true -> 0.25 : (s'=102) + 0.25 : (s'=117) + 0.5 : (s'=118);
	[] (s= 118) = true -> 0.0741398446171 : (s'=101) + 0.413762486127 : (s'=102) + 0.042397336293 : (s'=117) + 0.469700332963 : (s'=118);
	[] (s= 119) = true -> 0.789473684211 : (s'=119) + 0.210526315789 : (s'=120);
	[] (s= 120) = true -> 0.620236686391 : (s'=120) + 0.217041420118 : (s'=121) + 0.151952662722 : (s'=136) + 0.0107692307692 : (s'=137);
	[] (s= 121) = true -> 0.392857142857 : (s'=121) + 0.0357142857143 : (s'=136) + 0.571428571429 : (s'=137);
	[] (s= 122) = true -> 0.151133501259 : (s'=122) + 0.00503778337531 : (s'=123) + 0.400503778338 : (s'=138) + 0.443324937028 : (s'=139);
	[] (s= 123) = true -> 1.00031094519E-13 : (s'=123) + 0.870036101083 : (s'=139) + 0.0768437338834 : (s'=140) + 0.0422898401238 : (s'=155) + 0.0108303249097 : (s'=156);
	[] (s= 124) = true -> 0.362894144144 : (s'=140) + 0.345720720721 : (s'=141) + 0.212274774775 : (s'=156) + 0.0791103603604 : (s'=157);
	[] (s= 125) = true -> 0.29356223176 : (s'=141) + 0.70643776824 : (s'=157);
	[] (s= 126) = true -> 1.0 : (s'=126);
	[] (s= 127) = true -> 1.0 : (s'=127);
	[] (s= 128) = true -> 1.0 : (s'=128);
	[] (s= 129) = true -> 0.125 : (s'=97) + 0.75 : (s'=98) + 0.125 : (s'=114);
	[] (s= 130) = true -> 0.340425531915 : (s'=98) + 0.38829787234 : (s'=99) + 0.223404255319 : (s'=114) + 0.0478723404255 : (s'=115) + 5.00044450291E-13 : (s'=130);
	[] (s= 131) = true -> 0.36 : (s'=99) + 0.08 : (s'=114) + 0.56 : (s'=115);
	[] (s= 132) = true -> 0.0294117647059 : (s'=100) + 0.279411764706 : (s'=115) + 0.691176470588 : (s'=116) + 9.99200722163E-14 : (s'=132);
	[] (s= 133) = true -> 0.679147165887 : (s'=116) + 0.0946437857514 : (s'=117) + 0.0275611024441 : (s'=132) + 0.198647945918 : (s'=133);
	[] (s= 134) = true -> 0.277777777778 : (s'=118) + 0.222222222222 : (s'=119) + 0.388888888889 : (s'=134) + 0.111111111111 : (s'=135);
	[] (s= 135) = true -> 0.269967304998 : (s'=118) + 0.143858010276 : (s'=119) + 0.216487622606 : (s'=134) + 0.369687062121 : (s'=135);
	[] (s= 136) = true -> 0.0416319733555 : (s'=119) + 0.953788509575 : (s'=135) + 4.12303829712E-13 : (s'=136) + 0.00457951706911 : (s'=152);
	[] (s= 137) = true -> 0.484095658231 : (s'=137) + 0.00139308103088 : (s'=138) + 0.514279080567 : (s'=153) + 2.32180171813E-4 : (s'=154);
	[] (s= 138) = true -> 0.296 : (s'=138) + 0.7 : (s'=154) + 0.004 : (s'=155);
	[] (s= 139) = true -> 0.920090293454 : (s'=155) + 0.0419864559819 : (s'=156) + 0.0379232505643 : (s'=171);
	[] (s= 140) = true -> 0.175587467363 : (s'=156) + 0.0065274151436 : (s'=157) + 0.00195822454308 : (s'=171) + 0.810704960836 : (s'=172) + 0.00522193211488 : (s'=173);
	[] (s= 141) = true -> 0.199555555556 : (s'=157) + 0.0648888888888 : (s'=172) + 0.735555555555 : (s'=173);
	[] (s= 142) = true -> 1.0 : (s'=142);
	[] (s= 143) = true -> 1.0 : (s'=143);
	[] (s= 144) = true -> 1.0 : (s'=144);
	[] (s= 145) = true -> 1.0 : (s'=113);
	[] (s= 146) = true -> 1.0 : (s'=114);
	[] (s= 147) = true -> 0.388888888889 : (s'=115) + 0.0111111111111 : (s'=116) + 0.0111111111111 : (s'=130) + 0.577777777778 : (s'=131) + 0.0111111111111 : (s'=132);
	[] (s= 148) = true -> 0.0769230769231 : (s'=115) + 0.0961538461538 : (s'=116) + 0.451923076923 : (s'=131) + 0.375 : (s'=132) + 9.99200722163E-14 : (s'=148);
	[] (s= 149) = true -> 0.1 : (s'=132) + 0.9 : (s'=133);
	[] (s= 150) = true -> 0.293698630137 : (s'=133) + 0.0252054794521 : (s'=134) + 0.681095890411 : (s'=149) + 3.09959364111E-13 : (s'=150);
	[] (s= 151) = true -> 0.16 : (s'=134) + 0.226666666667 : (s'=135) + 0.1 : (s'=150) + 0.513333333333 : (s'=151);
	[] (s= 152) = true -> 0.142857142857 : (s'=135) + 0.857142857142 : (s'=151) + 4.79971618006E-13 : (s'=152);
	[] (s= 153) = true -> 0.0243341636329 : (s'=152) + 0.542824295842 : (s'=153) + 0.0421536692853 : (s'=168) + 0.39068787124 : (s'=169);
	[] (s= 154) = true -> 0.30487804878 : (s'=153) + 0.0609756097561 : (s'=154) + 0.487804878049 : (s'=169) + 0.146341463415 : (s'=170);
	[] (s= 155) = true -> 0.0290556900726 : (s'=170) + 0.965133171913 : (s'=171) + 0.00581113801453 : (s'=187);
	[] (s= 156) = true -> 1.99951166735E-13 : (s'=156) + 0.0889464594128 : (s'=171) + 0.759067357513 : (s'=172) + 0.027633851468 : (s'=187) + 0.124352331606 : (s'=188);
	[] (s= 157) = true -> 0.0670136629798 : (s'=172) + 0.601821730644 : (s'=173) + 0.0865322055953 : (s'=188) + 0.244632400781 : (s'=189);
	[] (s= 158) = true -> 1.0 : (s'=158);
	[] (s= 159) = true -> 1.0 : (s'=159);
	[] (s= 160) = true -> 1.0 : (s'=160);
	[] (s= 161) = true -> 1.0 : (s'=129);
	[] (s= 162) = true -> 0.746268656716 : (s'=130) + 0.253731343284 : (s'=146);
	[] (s= 163) = true -> 0.359375 : (s'=130) + 0.1875 : (s'=131) + 0.328125 : (s'=146) + 0.125 : (s'=147);
	[] (s= 164) = true -> 0.197674418605 : (s'=131) + 0.616279069767 : (s'=147) + 0.186046511628 : (s'=148);
	[] (s= 165) = true -> 0.571428571429 : (s'=148) + 0.25 : (s'=149) + 0.0714285714286 : (s'=164) + 0.107142857143 : (s'=165);
	[] (s= 166) = true -> 0.422622747048 : (s'=149) + 0.451211932878 : (s'=150) + 0.0385332504661 : (s'=165) + 0.0876320696084 : (s'=166);
	[] (s= 167) = true -> 0.294442638934 : (s'=150) + 3.24991875203E-4 : (s'=151) + 0.446213844654 : (s'=166) + 0.259018524537 : (s'=167);
	[] (s= 168) = true -> 0.0217723453018 : (s'=151) + 0.875095492743 : (s'=167) + 0.0996944232238 : (s'=168) + 0.00114591291062 : (s'=183) + 0.00229182582124 : (s'=184);
	[] (s= 169) = true -> 0.254807692308 : (s'=168) + 0.288461538462 : (s'=169) + 0.254807692308 : (s'=184) + 0.201923076923 : (s'=185);
	[] (s= 170) = true -> 0.0858585858585 : (s'=169) + 0.212121212121 : (s'=170) + 0.19696969697 : (s'=185) + 0.505050505051 : (s'=186);
	[] (s= 171) = true -> 4.51467268623E-4 : (s'=170) + 0.00361173814898 : (s'=171) + 0.379683972912 : (s'=186) + 0.615801354402 : (s'=187) + 4.51467268623E-4 : (s'=203);
	[] (s= 172) = true -> 0.189446366782 : (s'=187) + 0.708477508651 : (s'=188) + 0.0402249134948 : (s'=203) + 0.0618512110727 : (s'=204);
	[] (s= 173) = true -> 0.42840329541 : (s'=188) + 0.151431934092 : (s'=189) + 0.293840721852 : (s'=204) + 0.126324048647 : (s'=205);
	[] (s= 174) = true -> 1.0 : (s'=174);
	[] (s= 175) = true -> 1.0 : (s'=175);
	[] (s= 176) = true -> 1.0 : (s'=176);
	[] (s= 177) = true -> 1.0 : (s'=177);
	[] (s= 178) = true -> 0.823529411765 : (s'=146) + 0.176470588235 : (s'=162);
	[] (s= 179) = true -> 0.111545988258 : (s'=146) + 0.207436399217 : (s'=147) + 0.0626223091977 : (s'=162) + 0.618395303327 : (s'=163) + 3.00093283556E-13 : (s'=179);
	[] (s= 180) = true -> 0.156626506024 : (s'=147) + 0.819277108434 : (s'=163) + 0.0240963855422 : (s'=164);
	[] (s= 181) = true -> 0.0309278350515 : (s'=148) + 0.876288659794 : (s'=164) + 0.0412371134021 : (s'=165) + 0.0515463917526 : (s'=180);
	[] (s= 182) = true -> 0.636904761905 : (s'=165) + 0.0535714285714 : (s'=166) + 0.309523809524 : (s'=181);
	[] (s= 183) = true -> 0.25 : (s'=182) + 0.75 : (s'=183);
	[] (s= 184) = true -> 0.666666666667 : (s'=183) + 0.333333333333 : (s'=184);
	[] (s= 185) = true -> 0.360824742268 : (s'=184) + 0.371134020619 : (s'=185) + 0.144329896907 : (s'=200) + 0.123711340206 : (s'=201);
	[] (s= 186) = true -> 0.0645161290323 : (s'=185) + 0.219941348974 : (s'=186) + 0.299120234604 : (s'=201) + 0.41642228739 : (s'=202);
	[] (s= 187) = true -> 0.00211416490486 : (s'=186) + 6.02376969071E-14 : (s'=187) + 0.476744186047 : (s'=202) + 0.505813953488 : (s'=203) + 0.0153276955603 : (s'=219);
	[] (s= 188) = true -> 0.425321463897 : (s'=203) + 0.160567095285 : (s'=204) + 0.121002307946 : (s'=219) + 0.293109132872 : (s'=220);
	[] (s= 189) = true -> 0.454545454545 : (s'=204) + 0.545454545455 : (s'=220);
	[] (s= 190) = true -> 1.0 : (s'=190);
	[] (s= 191) = true -> 1.0 : (s'=191);
	[] (s= 192) = true -> 1.0 : (s'=192);
	[] (s= 193) = true -> 1.0 : (s'=193);
	[] (s= 194) = true -> 1.0 : (s'=162);
	[] (s= 195) = true -> 0.175 : (s'=162) + 0.475 : (s'=163) + 0.0428571428571 : (s'=178) + 0.307142857143 : (s'=179);
	[] (s= 196) = true -> 0.125 : (s'=163) + 0.0833333333333 : (s'=164) + 0.375 : (s'=179) + 0.416666666667 : (s'=180);
	[] (s= 197) = true -> 0.333333333333 : (s'=180) + 0.666666666667 : (s'=181);
	[] (s= 198) = true -> 0.0217391304348 : (s'=180) + 0.565217391304 : (s'=181) + 0.0108695652174 : (s'=196) + 0.402173913043 : (s'=197) + 8.00026711545E-13 : (s'=198);
	[] (s= 199) = true -> 0.135338345865 : (s'=182) + 0.00751879699248 : (s'=197) + 0.857142857143 : (s'=198);
	[] (s= 200) = true -> 0.428571428571 : (s'=199) + 0.571428571429 : (s'=200);
	[] (s= 201) = true -> 0.27397260274 : (s'=200) + 0.287671232877 : (s'=201) + 0.164383561644 : (s'=216) + 0.27397260274 : (s'=217);
	[] (s= 202) = true -> 0.0669417698304 : (s'=201) + 0.198532783127 : (s'=202) + 0.315451627694 : (s'=217) + 0.419073819349 : (s'=218);
	[] (s= 203) = true -> 0.0289855072464 : (s'=202) + 0.304347826087 : (s'=218) + 0.666666666667 : (s'=219);
	[] (s= 204) = true -> 5.25024468345E-13 : (s'=204) + 0.392920353982 : (s'=219) + 0.341592920354 : (s'=220) + 0.264896755162 : (s'=235) + 5.89970501475E-4 : (s'=236);
	[] (s= 205) = true -> 0.912878787879 : (s'=220) + 0.0795454545455 : (s'=221) + 0.00757575757576 : (s'=236);
	[] (s= 206) = true -> 1.0 : (s'=206);
	[] (s= 207) = true -> 1.0 : (s'=207);
	[] (s= 208) = true -> 1.0 : (s'=208);
	[] (s= 209) = true -> 1.0 : (s'=209);
	[] (s= 210) = true -> 1.0 : (s'=210);
	[] (s= 211) = true -> 1.0 : (s'=178);
	[] (s= 212) = true -> 0.209770114943 : (s'=179) + 0.0459770114943 : (s'=180) + 0.326149425287 : (s'=195) + 0.418103448276 : (s'=196);
	[] (s= 213) = true -> 0.590909090909 : (s'=196) + 0.386363636364 : (s'=197) + 0.0227272727273 : (s'=213);
	[] (s= 214) = true -> 0.375 : (s'=197) + 0.25 : (s'=198) + 0.166666666667 : (s'=213) + 0.208333333333 : (s'=214);
	[] (s= 215) = true -> 0.00247524752475 : (s'=197) + 0.111386138614 : (s'=198) + 0.0519801980198 : (s'=213) + 0.834158415842 : (s'=214);
	[] (s= 216) = true -> 0.5 : (s'=215) + 0.451612903226 : (s'=216) + 0.0483870967742 : (s'=232);
	[] (s= 217) = true -> 0.110282574568 : (s'=216) + 0.332025117739 : (s'=217) + 0.271193092622 : (s'=232) + 0.286499215071 : (s'=233);
	[] (s= 218) = true -> 0.518793706294 : (s'=217) + 0.0157342657343 : (s'=218) + 0.170454545455 : (s'=233) + 0.295017482517 : (s'=234);
	[] (s= 219) = true -> 0.0480769230769 : (s'=218) + 0.423076923077 : (s'=234) + 0.519230769231 : (s'=235) + 0.00961538461538 : (s'=251);
	[] (s= 220) = true -> 1.39888101103E-14 : (s'=220) + 0.771203155819 : (s'=235) + 0.168639053254 : (s'=236) + 9.86193293886E-4 : (s'=251) + 0.0591715976331 : (s'=252);
	[] (s= 221) = true -> 1.0 : (s'=252);
	[] (s= 222) = true -> 1.0 : (s'=222);
	[] (s= 223) = true -> 1.0 : (s'=223);
	[] (s= 224) = true -> 1.0 : (s'=224);
	[] (s= 225) = true -> 1.0 : (s'=225);
	[] (s= 226) = true -> 1.0 : (s'=226);
	[] (s= 227) = true -> 1.0 : (s'=227);
	[] (s= 228) = true -> 0.413043478261 : (s'=196) + 0.00724637681159 : (s'=211) + 0.579710144928 : (s'=212);
	[] (s= 229) = true -> 0.41935483871 : (s'=212) + 0.387096774194 : (s'=213) + 0.193548387097 : (s'=229);
	[] (s= 230) = true -> 0.363636363636 : (s'=213) + 0.318181818182 : (s'=214) + 0.181818181818 : (s'=229) + 0.136363636364 : (s'=230);
	[] (s= 231) = true -> 0.124183006536 : (s'=214) + 0.111111111111 : (s'=215) + 0.267973856209 : (s'=230) + 0.496732026144 : (s'=231);
	[] (s= 232) = true -> 0.04 : (s'=215) + 0.48 : (s'=231) + 0.48 : (s'=232);
	[] (s= 233) = true -> 0.5 : (s'=232) + 0.25 : (s'=248) + 0.25 : (s'=249);
	[] (s= 234) = true -> 0.149122807018 : (s'=233) + 0.078548644338 : (s'=234) + 0.635167464114 : (s'=249) + 0.13716108453 : (s'=250);
	[] (s= 235) = true -> 0.0584749111023 : (s'=234) + 0.131963650731 : (s'=250) + 0.804425128408 : (s'=251) + 0.00513630975899 : (s'=267);
	[] (s= 236) = true -> 5.99964522507E-13 : (s'=236) + 0.723618090452 : (s'=251) + 0.0100502512563 : (s'=252) + 0.190954773869 : (s'=267) + 0.0753768844221 : (s'=268);
	[] (s= 237) = true -> 1.0 : (s'=237);
	[] (s= 238) = true -> 1.0 : (s'=238);
	[] (s= 239) = true -> 1.0 : (s'=239);
	[] (s= 240) = true -> 1.0 : (s'=240);
	[] (s= 241) = true -> 1.0 : (s'=241);
	[] (s= 242) = true -> 1.0 : (s'=242);
	[] (s= 243) = true -> 1.0 : (s'=243);
	[] (s= 244) = true -> 1.0 : (s'=244);
	[] (s= 245) = true -> 0.103448275862 : (s'=228) + 0.827586206897 : (s'=229) + 0.0689655172414 : (s'=245);
	[] (s= 246) = true -> 0.0833333333333 : (s'=229) + 0.416666666667 : (s'=230) + 0.5 : (s'=246);
	[] (s= 247) = true -> 0.0899653979239 : (s'=230) + 0.0397923875433 : (s'=231) + 0.773356401384 : (s'=246) + 0.0968858131488 : (s'=247);
	[] (s= 248) = true -> 0.142857142857 : (s'=247) + 0.857142857143 : (s'=248);
	[] (s= 249) = true -> 0.615384615385 : (s'=248) + 0.230769230769 : (s'=249) + 0.153846153846 : (s'=264);
	[] (s= 250) = true -> 0.0574162679426 : (s'=249) + 0.0688995215311 : (s'=250) + 0.0564593301435 : (s'=265) + 0.817224880383 : (s'=266);
	[] (s= 251) = true -> 0.0234375 : (s'=250) + 1.00310384998E-13 : (s'=251) + 0.06640625 : (s'=266) + 0.900390625 : (s'=267) + 0.009765625 : (s'=283);
	[] (s= 252) = true -> 0.217391304348 : (s'=267) + 0.782608695652 : (s'=268);
	[] (s= 253) = true -> 1.0 : (s'=321);
	[] (s= 254) = true -> 1.0 : (s'=321);
	[] (s= 255) = true -> 1.0 : (s'=321);
	[] (s= 256) = true -> 1.0 : (s'=256);
	[] (s= 257) = true -> 1.0 : (s'=257);
	[] (s= 258) = true -> 1.0 : (s'=258);
	[] (s= 259) = true -> 1.0 : (s'=259);
	[] (s= 260) = true -> 1.0 : (s'=260);
	[] (s= 261) = true -> 1.0 : (s'=261);
	[] (s= 262) = true -> 0.433333333333 : (s'=245) + 0.166666666667 : (s'=246) + 0.4 : (s'=262);
	[] (s= 263) = true -> 0.0323054331865 : (s'=246) + 0.126284875184 : (s'=247) + 0.0011013215859 : (s'=262) + 0.834067547724 : (s'=263) + 0.00624082232012 : (s'=264);
	[] (s= 264) = true -> 0.125 : (s'=263) + 0.875 : (s'=264);
	[] (s= 265) = true -> 0.625 : (s'=264) + 0.125 : (s'=265) + 0.25 : (s'=281);
	[] (s= 266) = true -> 0.147058823529 : (s'=265) + 0.235294117647 : (s'=266) + 0.617647058824 : (s'=282);
	[] (s= 267) = true -> 0.111111111111 : (s'=266) + 0.111111111111 : (s'=267) + 0.111111111111 : (s'=282) + 0.555555555556 : (s'=283) + 0.111111111111 : (s'=299);
	[] (s= 268) = true -> 0.444444444444 : (s'=283) + 0.555555555556 : (s'=284);
	[] (s= 269) = true -> 1.0 : (s'=321);
	[] (s= 270) = true -> 1.0 : (s'=321);
	[] (s= 271) = true -> 1.0 : (s'=321);
	[] (s= 272) = true -> 1.0 : (s'=272);
	[] (s= 273) = true -> 1.0 : (s'=273);
	[] (s= 274) = true -> 1.0 : (s'=274);
	[] (s= 275) = true -> 1.0 : (s'=275);
	[] (s= 276) = true -> 1.0 : (s'=276);
	[] (s= 277) = true -> 1.0 : (s'=277);
	[] (s= 278) = true -> 0.5 : (s'=262) + 0.5 : (s'=263);
	[] (s= 279) = true -> 1.0 : (s'=279);
	[] (s= 280) = true -> 1.0 : (s'=280);
	[] (s= 281) = true -> 0.389035087719 : (s'=281) + 0.0339912280702 : (s'=282) + 0.264254385965 : (s'=297) + 0.312719298246 : (s'=298);
	[] (s= 282) = true -> 0.275 : (s'=282) + 0.425 : (s'=298) + 0.3 : (s'=299);
	[] (s= 283) = true -> 1.0 : (s'=299);
	[] (s= 284) = true -> 1.0 : (s'=300);
	[] (s= 285) = true -> 1.0 : (s'=321);
	[] (s= 286) = true -> 1.0 : (s'=321);
	[] (s= 287) = true -> 1.0 : (s'=321);
	[] (s= 288) = true -> 1.0 : (s'=288);
	[] (s= 289) = true -> 1.0 : (s'=289);
	[] (s= 290) = true -> 1.0 : (s'=290);
	[] (s= 291) = true -> 1.0 : (s'=291);
	[] (s= 292) = true -> 1.0 : (s'=292);
	[] (s= 293) = true -> 1.0 : (s'=293);
	[] (s= 294) = true -> 1.0 : (s'=294);
	[] (s= 295) = true -> 1.0 : (s'=295);
	[] (s= 296) = true -> 1.0 : (s'=296);
	[] (s= 297) = true -> 1.0 : (s'=297);
	[] (s= 298) = true -> 1.0 : (s'=298);
	[] (s= 299) = true -> 1.0 : (s'=299);
	[] (s= 300) = true -> 1.0 : (s'=300);
	[] (s= 301) = true -> 1.0 : (s'=321);
	[] (s= 302) = true -> 1.0 : (s'=321);
	[] (s= 303) = true -> 1.0 : (s'=321);
	[] (s= 304) = true -> 1.0 : (s'=304);
	[] (s= 305) = true -> 1.0 : (s'=305);
	[] (s= 306) = true -> 1.0 : (s'=306);
	[] (s= 307) = true -> 1.0 : (s'=307);
	[] (s= 308) = true -> 1.0 : (s'=308);
	[] (s= 309) = true -> 1.0 : (s'=309);
	[] (s= 310) = true -> 1.0 : (s'=310);
	[] (s= 311) = true -> 1.0 : (s'=311);
	[] (s= 312) = true -> 1.0 : (s'=312);
	[] (s= 313) = true -> 1.0 : (s'=313);
	[] (s= 314) = true -> 1.0 : (s'=314);
	[] (s= 315) = true -> 1.0 : (s'=315);
	[] (s= 316) = true -> 1.0 : (s'=316);
	[] (s= 317) = true -> 1.0 : (s'=321);
	[] (s= 318) = true -> 1.0 : (s'=321);
	[] (s= 319) = true -> 1.0 : (s'=321);
	[] (s= 320) = true -> 0.5 : (s'=120) + 0.5 : (s'=136);
	[] (s= 321) = true -> 1.0 : (s'=321);

endmodule

init s = 320 endinit

