mdp

const int states = 66;
const int actions = 5;

module grid_world

	s : [0..66];

	[a0] s=0 -> 1.0 : (s'=0);
	[a1] s=0 -> 0.88 : (s'=1) + 0.08 : (s'=8) + 0.040000000000000036 : (s'=0);
	[a2] s=0 -> 0.08 : (s'=1) + 0.88 : (s'=8) + 0.040000000000000036 : (s'=0);
	[a3] s=0 -> 0.88 : (s'=1) + 0.08 : (s'=8) + 0.040000000000000036 : (s'=0);
	[a4] s=0 -> 0.08 : (s'=1) + 0.88 : (s'=8) + 0.040000000000000036 : (s'=0);
	[a0] s=1 -> 1.0 : (s'=1);
	[a1] s=1 -> 0.04 : (s'=0) + 0.84 : (s'=2) + 0.08 : (s'=9) + 0.040000000000000036 : (s'=1);
	[a2] s=1 -> 0.04 : (s'=0) + 0.04 : (s'=2) + 0.88 : (s'=9) + 0.040000000000000036 : (s'=1);
	[a3] s=1 -> 0.84 : (s'=0) + 0.04 : (s'=2) + 0.08 : (s'=9) + 0.040000000000000036 : (s'=1);
	[a4] s=1 -> 0.04 : (s'=0) + 0.04 : (s'=2) + 0.88 : (s'=9) + 0.040000000000000036 : (s'=1);
	[a0] s=2 -> 1.0 : (s'=2);
	[a1] s=2 -> 0.04 : (s'=1) + 0.84 : (s'=3) + 0.08 : (s'=10) + 0.040000000000000036 : (s'=2);
	[a2] s=2 -> 0.04 : (s'=1) + 0.04 : (s'=3) + 0.88 : (s'=10) + 0.040000000000000036 : (s'=2);
	[a3] s=2 -> 0.84 : (s'=1) + 0.04 : (s'=3) + 0.08 : (s'=10) + 0.040000000000000036 : (s'=2);
	[a4] s=2 -> 0.04 : (s'=1) + 0.04 : (s'=3) + 0.88 : (s'=10) + 0.040000000000000036 : (s'=2);
	[a0] s=3 -> 1.0 : (s'=3);
	[a1] s=3 -> 0.04 : (s'=2) + 0.84 : (s'=4) + 0.08 : (s'=11) + 0.040000000000000036 : (s'=3);
	[a2] s=3 -> 0.04 : (s'=2) + 0.04 : (s'=4) + 0.88 : (s'=11) + 0.040000000000000036 : (s'=3);
	[a3] s=3 -> 0.84 : (s'=2) + 0.04 : (s'=4) + 0.08 : (s'=11) + 0.040000000000000036 : (s'=3);
	[a4] s=3 -> 0.04 : (s'=2) + 0.04 : (s'=4) + 0.88 : (s'=11) + 0.040000000000000036 : (s'=3);
	[a0] s=4 -> 1.0 : (s'=4);
	[a1] s=4 -> 0.04 : (s'=3) + 0.84 : (s'=5) + 0.08 : (s'=12) + 0.040000000000000036 : (s'=4);
	[a2] s=4 -> 0.04 : (s'=3) + 0.04 : (s'=5) + 0.88 : (s'=12) + 0.040000000000000036 : (s'=4);
	[a3] s=4 -> 0.84 : (s'=3) + 0.04 : (s'=5) + 0.08 : (s'=12) + 0.040000000000000036 : (s'=4);
	[a4] s=4 -> 0.04 : (s'=3) + 0.04 : (s'=5) + 0.88 : (s'=12) + 0.040000000000000036 : (s'=4);
	[a0] s=5 -> 1.0 : (s'=5);
	[a1] s=5 -> 0.04 : (s'=4) + 0.84 : (s'=6) + 0.08 : (s'=13) + 0.040000000000000036 : (s'=5);
	[a2] s=5 -> 0.04 : (s'=4) + 0.04 : (s'=6) + 0.88 : (s'=13) + 0.040000000000000036 : (s'=5);
	[a3] s=5 -> 0.84 : (s'=4) + 0.04 : (s'=6) + 0.08 : (s'=13) + 0.040000000000000036 : (s'=5);
	[a4] s=5 -> 0.04 : (s'=4) + 0.04 : (s'=6) + 0.88 : (s'=13) + 0.040000000000000036 : (s'=5);
	[a0] s=6 -> 1.0 : (s'=6);
	[a1] s=6 -> 0.04 : (s'=5) + 0.84 : (s'=7) + 0.08 : (s'=14) + 0.040000000000000036 : (s'=6);
	[a2] s=6 -> 0.04 : (s'=5) + 0.04 : (s'=7) + 0.88 : (s'=14) + 0.040000000000000036 : (s'=6);
	[a3] s=6 -> 0.84 : (s'=5) + 0.04 : (s'=7) + 0.08 : (s'=14) + 0.040000000000000036 : (s'=6);
	[a4] s=6 -> 0.04 : (s'=5) + 0.04 : (s'=7) + 0.88 : (s'=14) + 0.040000000000000036 : (s'=6);
	[a0] s=7 -> 1.0 : (s'=7);
	[a1] s=7 -> 0.88 : (s'=6) + 0.08 : (s'=15) + 0.040000000000000036 : (s'=7);
	[a2] s=7 -> 0.08 : (s'=6) + 0.88 : (s'=15) + 0.040000000000000036 : (s'=7);
	[a3] s=7 -> 0.88 : (s'=6) + 0.08 : (s'=15) + 0.040000000000000036 : (s'=7);
	[a4] s=7 -> 0.08 : (s'=6) + 0.88 : (s'=15) + 0.040000000000000036 : (s'=7);
	[a0] s=8 -> 1.0 : (s'=8);
	[a1] s=8 -> 0.04 : (s'=0) + 0.88 : (s'=9) + 0.04 : (s'=16) + 0.039999999999999925 : (s'=8);
	[a2] s=8 -> 0.04 : (s'=0) + 0.08 : (s'=9) + 0.84 : (s'=16) + 0.040000000000000036 : (s'=8);
	[a3] s=8 -> 0.04 : (s'=0) + 0.88 : (s'=9) + 0.04 : (s'=16) + 0.039999999999999925 : (s'=8);
	[a4] s=8 -> 0.84 : (s'=0) + 0.08 : (s'=9) + 0.04 : (s'=16) + 0.040000000000000036 : (s'=8);
	[a0] s=9 -> 1.0 : (s'=9);
	[a1] s=9 -> 0.04 : (s'=1) + 0.04 : (s'=8) + 0.84 : (s'=10) + 0.04 : (s'=17) + 0.040000000000000036 : (s'=9);
	[a2] s=9 -> 0.04 : (s'=1) + 0.04 : (s'=8) + 0.04 : (s'=10) + 0.84 : (s'=17) + 0.040000000000000036 : (s'=9);
	[a3] s=9 -> 0.04 : (s'=1) + 0.84 : (s'=8) + 0.04 : (s'=10) + 0.04 : (s'=17) + 0.039999999999999925 : (s'=9);
	[a4] s=9 -> 0.84 : (s'=1) + 0.04 : (s'=8) + 0.04 : (s'=10) + 0.04 : (s'=17) + 0.039999999999999925 : (s'=9);
	[a0] s=10 -> 1.0 : (s'=10);
	[a1] s=10 -> 0.04 : (s'=2) + 0.04 : (s'=9) + 0.84 : (s'=11) + 0.04 : (s'=18) + 0.040000000000000036 : (s'=10);
	[a2] s=10 -> 0.04 : (s'=2) + 0.04 : (s'=9) + 0.04 : (s'=11) + 0.84 : (s'=18) + 0.040000000000000036 : (s'=10);
	[a3] s=10 -> 0.04 : (s'=2) + 0.84 : (s'=9) + 0.04 : (s'=11) + 0.04 : (s'=18) + 0.039999999999999925 : (s'=10);
	[a4] s=10 -> 0.84 : (s'=2) + 0.04 : (s'=9) + 0.04 : (s'=11) + 0.04 : (s'=18) + 0.039999999999999925 : (s'=10);
	[a0] s=11 -> 1.0 : (s'=11);
	[a1] s=11 -> 0.04 : (s'=3) + 0.04 : (s'=10) + 0.84 : (s'=12) + 0.04 : (s'=19) + 0.040000000000000036 : (s'=11);
	[a2] s=11 -> 0.04 : (s'=3) + 0.04 : (s'=10) + 0.04 : (s'=12) + 0.84 : (s'=19) + 0.040000000000000036 : (s'=11);
	[a3] s=11 -> 0.04 : (s'=3) + 0.84 : (s'=10) + 0.04 : (s'=12) + 0.04 : (s'=19) + 0.039999999999999925 : (s'=11);
	[a4] s=11 -> 0.84 : (s'=3) + 0.04 : (s'=10) + 0.04 : (s'=12) + 0.04 : (s'=19) + 0.039999999999999925 : (s'=11);
	[a0] s=12 -> 1.0 : (s'=12);
	[a1] s=12 -> 0.04 : (s'=4) + 0.04 : (s'=11) + 0.84 : (s'=13) + 0.04 : (s'=20) + 0.040000000000000036 : (s'=12);
	[a2] s=12 -> 0.04 : (s'=4) + 0.04 : (s'=11) + 0.04 : (s'=13) + 0.84 : (s'=20) + 0.040000000000000036 : (s'=12);
	[a3] s=12 -> 0.04 : (s'=4) + 0.84 : (s'=11) + 0.04 : (s'=13) + 0.04 : (s'=20) + 0.039999999999999925 : (s'=12);
	[a4] s=12 -> 0.84 : (s'=4) + 0.04 : (s'=11) + 0.04 : (s'=13) + 0.04 : (s'=20) + 0.039999999999999925 : (s'=12);
	[a0] s=13 -> 1.0 : (s'=13);
	[a1] s=13 -> 0.04 : (s'=5) + 0.04 : (s'=12) + 0.84 : (s'=14) + 0.04 : (s'=21) + 0.040000000000000036 : (s'=13);
	[a2] s=13 -> 0.04 : (s'=5) + 0.04 : (s'=12) + 0.04 : (s'=14) + 0.84 : (s'=21) + 0.040000000000000036 : (s'=13);
	[a3] s=13 -> 0.04 : (s'=5) + 0.84 : (s'=12) + 0.04 : (s'=14) + 0.04 : (s'=21) + 0.039999999999999925 : (s'=13);
	[a4] s=13 -> 0.84 : (s'=5) + 0.04 : (s'=12) + 0.04 : (s'=14) + 0.04 : (s'=21) + 0.039999999999999925 : (s'=13);
	[a0] s=14 -> 1.0 : (s'=14);
	[a1] s=14 -> 0.04 : (s'=6) + 0.04 : (s'=13) + 0.84 : (s'=15) + 0.04 : (s'=22) + 0.040000000000000036 : (s'=14);
	[a2] s=14 -> 0.04 : (s'=6) + 0.04 : (s'=13) + 0.04 : (s'=15) + 0.84 : (s'=22) + 0.040000000000000036 : (s'=14);
	[a3] s=14 -> 0.04 : (s'=6) + 0.84 : (s'=13) + 0.04 : (s'=15) + 0.04 : (s'=22) + 0.039999999999999925 : (s'=14);
	[a4] s=14 -> 0.84 : (s'=6) + 0.04 : (s'=13) + 0.04 : (s'=15) + 0.04 : (s'=22) + 0.039999999999999925 : (s'=14);
	[a0] s=15 -> 1.0 : (s'=15);
	[a1] s=15 -> 0.04 : (s'=7) + 0.88 : (s'=14) + 0.04 : (s'=23) + 0.039999999999999925 : (s'=15);
	[a2] s=15 -> 0.04 : (s'=7) + 0.08 : (s'=14) + 0.84 : (s'=23) + 0.040000000000000036 : (s'=15);
	[a3] s=15 -> 0.04 : (s'=7) + 0.88 : (s'=14) + 0.04 : (s'=23) + 0.039999999999999925 : (s'=15);
	[a4] s=15 -> 0.84 : (s'=7) + 0.08 : (s'=14) + 0.04 : (s'=23) + 0.040000000000000036 : (s'=15);
	[a0] s=16 -> 1.0 : (s'=16);
	[a1] s=16 -> 0.04 : (s'=8) + 0.88 : (s'=17) + 0.04 : (s'=24) + 0.039999999999999925 : (s'=16);
	[a2] s=16 -> 0.04 : (s'=8) + 0.08 : (s'=17) + 0.84 : (s'=24) + 0.040000000000000036 : (s'=16);
	[a3] s=16 -> 0.04 : (s'=8) + 0.88 : (s'=17) + 0.04 : (s'=24) + 0.039999999999999925 : (s'=16);
	[a4] s=16 -> 0.84 : (s'=8) + 0.08 : (s'=17) + 0.04 : (s'=24) + 0.040000000000000036 : (s'=16);
	[a0] s=17 -> 1.0 : (s'=17);
	[a1] s=17 -> 0.04 : (s'=9) + 0.04 : (s'=16) + 0.84 : (s'=18) + 0.04 : (s'=25) + 0.040000000000000036 : (s'=17);
	[a2] s=17 -> 0.04 : (s'=9) + 0.04 : (s'=16) + 0.04 : (s'=18) + 0.84 : (s'=25) + 0.040000000000000036 : (s'=17);
	[a3] s=17 -> 0.04 : (s'=9) + 0.84 : (s'=16) + 0.04 : (s'=18) + 0.04 : (s'=25) + 0.039999999999999925 : (s'=17);
	[a4] s=17 -> 0.84 : (s'=9) + 0.04 : (s'=16) + 0.04 : (s'=18) + 0.04 : (s'=25) + 0.039999999999999925 : (s'=17);
	[a0] s=18 -> 1.0 : (s'=18);
	[a1] s=18 -> 0.04 : (s'=10) + 0.04 : (s'=17) + 0.84 : (s'=19) + 0.04 : (s'=26) + 0.040000000000000036 : (s'=18);
	[a2] s=18 -> 0.04 : (s'=10) + 0.04 : (s'=17) + 0.04 : (s'=19) + 0.84 : (s'=26) + 0.040000000000000036 : (s'=18);
	[a3] s=18 -> 0.04 : (s'=10) + 0.84 : (s'=17) + 0.04 : (s'=19) + 0.04 : (s'=26) + 0.039999999999999925 : (s'=18);
	[a4] s=18 -> 0.84 : (s'=10) + 0.04 : (s'=17) + 0.04 : (s'=19) + 0.04 : (s'=26) + 0.039999999999999925 : (s'=18);
	[a0] s=19 -> 1.0 : (s'=19);
	[a1] s=19 -> 0.04 : (s'=11) + 0.04 : (s'=18) + 0.84 : (s'=20) + 0.04 : (s'=27) + 0.040000000000000036 : (s'=19);
	[a2] s=19 -> 0.04 : (s'=11) + 0.04 : (s'=18) + 0.04 : (s'=20) + 0.84 : (s'=27) + 0.040000000000000036 : (s'=19);
	[a3] s=19 -> 0.04 : (s'=11) + 0.84 : (s'=18) + 0.04 : (s'=20) + 0.04 : (s'=27) + 0.039999999999999925 : (s'=19);
	[a4] s=19 -> 0.84 : (s'=11) + 0.04 : (s'=18) + 0.04 : (s'=20) + 0.04 : (s'=27) + 0.039999999999999925 : (s'=19);
	[a0] s=20 -> 1.0 : (s'=20);
	[a1] s=20 -> 0.04 : (s'=12) + 0.04 : (s'=19) + 0.84 : (s'=21) + 0.04 : (s'=28) + 0.040000000000000036 : (s'=20);
	[a2] s=20 -> 0.04 : (s'=12) + 0.04 : (s'=19) + 0.04 : (s'=21) + 0.84 : (s'=28) + 0.040000000000000036 : (s'=20);
	[a3] s=20 -> 0.04 : (s'=12) + 0.84 : (s'=19) + 0.04 : (s'=21) + 0.04 : (s'=28) + 0.039999999999999925 : (s'=20);
	[a4] s=20 -> 0.84 : (s'=12) + 0.04 : (s'=19) + 0.04 : (s'=21) + 0.04 : (s'=28) + 0.039999999999999925 : (s'=20);
	[a0] s=21 -> 1.0 : (s'=21);
	[a1] s=21 -> 0.04 : (s'=13) + 0.04 : (s'=20) + 0.84 : (s'=22) + 0.04 : (s'=29) + 0.040000000000000036 : (s'=21);
	[a2] s=21 -> 0.04 : (s'=13) + 0.04 : (s'=20) + 0.04 : (s'=22) + 0.84 : (s'=29) + 0.040000000000000036 : (s'=21);
	[a3] s=21 -> 0.04 : (s'=13) + 0.84 : (s'=20) + 0.04 : (s'=22) + 0.04 : (s'=29) + 0.039999999999999925 : (s'=21);
	[a4] s=21 -> 0.84 : (s'=13) + 0.04 : (s'=20) + 0.04 : (s'=22) + 0.04 : (s'=29) + 0.039999999999999925 : (s'=21);
	[a0] s=22 -> 1.0 : (s'=22);
	[a1] s=22 -> 0.04 : (s'=14) + 0.04 : (s'=21) + 0.84 : (s'=23) + 0.04 : (s'=30) + 0.040000000000000036 : (s'=22);
	[a2] s=22 -> 0.04 : (s'=14) + 0.04 : (s'=21) + 0.04 : (s'=23) + 0.84 : (s'=30) + 0.040000000000000036 : (s'=22);
	[a3] s=22 -> 0.04 : (s'=14) + 0.84 : (s'=21) + 0.04 : (s'=23) + 0.04 : (s'=30) + 0.039999999999999925 : (s'=22);
	[a4] s=22 -> 0.84 : (s'=14) + 0.04 : (s'=21) + 0.04 : (s'=23) + 0.04 : (s'=30) + 0.039999999999999925 : (s'=22);
	[a0] s=23 -> 1.0 : (s'=23);
	[a1] s=23 -> 0.04 : (s'=15) + 0.88 : (s'=22) + 0.04 : (s'=31) + 0.039999999999999925 : (s'=23);
	[a2] s=23 -> 0.04 : (s'=15) + 0.08 : (s'=22) + 0.84 : (s'=31) + 0.040000000000000036 : (s'=23);
	[a3] s=23 -> 0.04 : (s'=15) + 0.88 : (s'=22) + 0.04 : (s'=31) + 0.039999999999999925 : (s'=23);
	[a4] s=23 -> 0.84 : (s'=15) + 0.08 : (s'=22) + 0.04 : (s'=31) + 0.040000000000000036 : (s'=23);
	[a0] s=24 -> 1.0 : (s'=24);
	[a1] s=24 -> 0.04 : (s'=16) + 0.88 : (s'=25) + 0.04 : (s'=32) + 0.039999999999999925 : (s'=24);
	[a2] s=24 -> 0.04 : (s'=16) + 0.08 : (s'=25) + 0.84 : (s'=32) + 0.040000000000000036 : (s'=24);
	[a3] s=24 -> 0.04 : (s'=16) + 0.88 : (s'=25) + 0.04 : (s'=32) + 0.039999999999999925 : (s'=24);
	[a4] s=24 -> 0.84 : (s'=16) + 0.08 : (s'=25) + 0.04 : (s'=32) + 0.040000000000000036 : (s'=24);
	[a0] s=25 -> 1.0 : (s'=25);
	[a1] s=25 -> 0.04 : (s'=17) + 0.04 : (s'=24) + 0.84 : (s'=26) + 0.04 : (s'=33) + 0.040000000000000036 : (s'=25);
	[a2] s=25 -> 0.04 : (s'=17) + 0.04 : (s'=24) + 0.04 : (s'=26) + 0.84 : (s'=33) + 0.040000000000000036 : (s'=25);
	[a3] s=25 -> 0.04 : (s'=17) + 0.84 : (s'=24) + 0.04 : (s'=26) + 0.04 : (s'=33) + 0.039999999999999925 : (s'=25);
	[a4] s=25 -> 0.84 : (s'=17) + 0.04 : (s'=24) + 0.04 : (s'=26) + 0.04 : (s'=33) + 0.039999999999999925 : (s'=25);
	[a0] s=26 -> 1.0 : (s'=26);
	[a1] s=26 -> 0.04 : (s'=18) + 0.04 : (s'=25) + 0.84 : (s'=27) + 0.04 : (s'=34) + 0.040000000000000036 : (s'=26);
	[a2] s=26 -> 0.04 : (s'=18) + 0.04 : (s'=25) + 0.04 : (s'=27) + 0.84 : (s'=34) + 0.040000000000000036 : (s'=26);
	[a3] s=26 -> 0.04 : (s'=18) + 0.84 : (s'=25) + 0.04 : (s'=27) + 0.04 : (s'=34) + 0.039999999999999925 : (s'=26);
	[a4] s=26 -> 0.84 : (s'=18) + 0.04 : (s'=25) + 0.04 : (s'=27) + 0.04 : (s'=34) + 0.039999999999999925 : (s'=26);
	[a0] s=27 -> 1.0 : (s'=27);
	[a1] s=27 -> 0.04 : (s'=19) + 0.04 : (s'=26) + 0.84 : (s'=28) + 0.04 : (s'=35) + 0.040000000000000036 : (s'=27);
	[a2] s=27 -> 0.04 : (s'=19) + 0.04 : (s'=26) + 0.04 : (s'=28) + 0.84 : (s'=35) + 0.040000000000000036 : (s'=27);
	[a3] s=27 -> 0.04 : (s'=19) + 0.84 : (s'=26) + 0.04 : (s'=28) + 0.04 : (s'=35) + 0.039999999999999925 : (s'=27);
	[a4] s=27 -> 0.84 : (s'=19) + 0.04 : (s'=26) + 0.04 : (s'=28) + 0.04 : (s'=35) + 0.039999999999999925 : (s'=27);
	[a0] s=28 -> 1.0 : (s'=28);
	[a1] s=28 -> 0.04 : (s'=20) + 0.04 : (s'=27) + 0.84 : (s'=29) + 0.04 : (s'=36) + 0.040000000000000036 : (s'=28);
	[a2] s=28 -> 0.04 : (s'=20) + 0.04 : (s'=27) + 0.04 : (s'=29) + 0.84 : (s'=36) + 0.040000000000000036 : (s'=28);
	[a3] s=28 -> 0.04 : (s'=20) + 0.84 : (s'=27) + 0.04 : (s'=29) + 0.04 : (s'=36) + 0.039999999999999925 : (s'=28);
	[a4] s=28 -> 0.84 : (s'=20) + 0.04 : (s'=27) + 0.04 : (s'=29) + 0.04 : (s'=36) + 0.039999999999999925 : (s'=28);
	[a0] s=29 -> 1.0 : (s'=29);
	[a1] s=29 -> 0.04 : (s'=21) + 0.04 : (s'=28) + 0.84 : (s'=30) + 0.04 : (s'=37) + 0.040000000000000036 : (s'=29);
	[a2] s=29 -> 0.04 : (s'=21) + 0.04 : (s'=28) + 0.04 : (s'=30) + 0.84 : (s'=37) + 0.040000000000000036 : (s'=29);
	[a3] s=29 -> 0.04 : (s'=21) + 0.84 : (s'=28) + 0.04 : (s'=30) + 0.04 : (s'=37) + 0.039999999999999925 : (s'=29);
	[a4] s=29 -> 0.84 : (s'=21) + 0.04 : (s'=28) + 0.04 : (s'=30) + 0.04 : (s'=37) + 0.039999999999999925 : (s'=29);
	[a0] s=30 -> 1.0 : (s'=30);
	[a1] s=30 -> 0.04 : (s'=22) + 0.04 : (s'=29) + 0.84 : (s'=31) + 0.04 : (s'=38) + 0.040000000000000036 : (s'=30);
	[a2] s=30 -> 0.04 : (s'=22) + 0.04 : (s'=29) + 0.04 : (s'=31) + 0.84 : (s'=38) + 0.040000000000000036 : (s'=30);
	[a3] s=30 -> 0.04 : (s'=22) + 0.84 : (s'=29) + 0.04 : (s'=31) + 0.04 : (s'=38) + 0.039999999999999925 : (s'=30);
	[a4] s=30 -> 0.84 : (s'=22) + 0.04 : (s'=29) + 0.04 : (s'=31) + 0.04 : (s'=38) + 0.039999999999999925 : (s'=30);
	[a0] s=31 -> 1.0 : (s'=31);
	[a1] s=31 -> 0.04 : (s'=23) + 0.88 : (s'=30) + 0.04 : (s'=39) + 0.039999999999999925 : (s'=31);
	[a2] s=31 -> 0.04 : (s'=23) + 0.08 : (s'=30) + 0.84 : (s'=39) + 0.040000000000000036 : (s'=31);
	[a3] s=31 -> 0.04 : (s'=23) + 0.88 : (s'=30) + 0.04 : (s'=39) + 0.039999999999999925 : (s'=31);
	[a4] s=31 -> 0.84 : (s'=23) + 0.08 : (s'=30) + 0.04 : (s'=39) + 0.040000000000000036 : (s'=31);
	[a0] s=32 -> 1.0 : (s'=32);
	[a1] s=32 -> 0.04 : (s'=24) + 0.88 : (s'=33) + 0.04 : (s'=40) + 0.039999999999999925 : (s'=32);
	[a2] s=32 -> 0.04 : (s'=24) + 0.08 : (s'=33) + 0.84 : (s'=40) + 0.040000000000000036 : (s'=32);
	[a3] s=32 -> 0.04 : (s'=24) + 0.88 : (s'=33) + 0.04 : (s'=40) + 0.039999999999999925 : (s'=32);
	[a4] s=32 -> 0.84 : (s'=24) + 0.08 : (s'=33) + 0.04 : (s'=40) + 0.040000000000000036 : (s'=32);
	[a0] s=33 -> 1.0 : (s'=33);
	[a1] s=33 -> 0.04 : (s'=25) + 0.04 : (s'=32) + 0.84 : (s'=34) + 0.04 : (s'=41) + 0.040000000000000036 : (s'=33);
	[a2] s=33 -> 0.04 : (s'=25) + 0.04 : (s'=32) + 0.04 : (s'=34) + 0.84 : (s'=41) + 0.040000000000000036 : (s'=33);
	[a3] s=33 -> 0.04 : (s'=25) + 0.84 : (s'=32) + 0.04 : (s'=34) + 0.04 : (s'=41) + 0.039999999999999925 : (s'=33);
	[a4] s=33 -> 0.84 : (s'=25) + 0.04 : (s'=32) + 0.04 : (s'=34) + 0.04 : (s'=41) + 0.039999999999999925 : (s'=33);
	[a0] s=34 -> 1.0 : (s'=34);
	[a1] s=34 -> 0.04 : (s'=26) + 0.04 : (s'=33) + 0.84 : (s'=35) + 0.04 : (s'=42) + 0.040000000000000036 : (s'=34);
	[a2] s=34 -> 0.04 : (s'=26) + 0.04 : (s'=33) + 0.04 : (s'=35) + 0.84 : (s'=42) + 0.040000000000000036 : (s'=34);
	[a3] s=34 -> 0.04 : (s'=26) + 0.84 : (s'=33) + 0.04 : (s'=35) + 0.04 : (s'=42) + 0.039999999999999925 : (s'=34);
	[a4] s=34 -> 0.84 : (s'=26) + 0.04 : (s'=33) + 0.04 : (s'=35) + 0.04 : (s'=42) + 0.039999999999999925 : (s'=34);
	[a0] s=35 -> 1.0 : (s'=35);
	[a1] s=35 -> 0.04 : (s'=27) + 0.04 : (s'=34) + 0.84 : (s'=36) + 0.04 : (s'=43) + 0.040000000000000036 : (s'=35);
	[a2] s=35 -> 0.04 : (s'=27) + 0.04 : (s'=34) + 0.04 : (s'=36) + 0.84 : (s'=43) + 0.040000000000000036 : (s'=35);
	[a3] s=35 -> 0.04 : (s'=27) + 0.84 : (s'=34) + 0.04 : (s'=36) + 0.04 : (s'=43) + 0.039999999999999925 : (s'=35);
	[a4] s=35 -> 0.84 : (s'=27) + 0.04 : (s'=34) + 0.04 : (s'=36) + 0.04 : (s'=43) + 0.039999999999999925 : (s'=35);
	[a0] s=36 -> 1.0 : (s'=36);
	[a1] s=36 -> 0.04 : (s'=28) + 0.04 : (s'=35) + 0.84 : (s'=37) + 0.04 : (s'=44) + 0.040000000000000036 : (s'=36);
	[a2] s=36 -> 0.04 : (s'=28) + 0.04 : (s'=35) + 0.04 : (s'=37) + 0.84 : (s'=44) + 0.040000000000000036 : (s'=36);
	[a3] s=36 -> 0.04 : (s'=28) + 0.84 : (s'=35) + 0.04 : (s'=37) + 0.04 : (s'=44) + 0.039999999999999925 : (s'=36);
	[a4] s=36 -> 0.84 : (s'=28) + 0.04 : (s'=35) + 0.04 : (s'=37) + 0.04 : (s'=44) + 0.039999999999999925 : (s'=36);
	[a0] s=37 -> 1.0 : (s'=37);
	[a1] s=37 -> 0.04 : (s'=29) + 0.04 : (s'=36) + 0.84 : (s'=38) + 0.04 : (s'=45) + 0.040000000000000036 : (s'=37);
	[a2] s=37 -> 0.04 : (s'=29) + 0.04 : (s'=36) + 0.04 : (s'=38) + 0.84 : (s'=45) + 0.040000000000000036 : (s'=37);
	[a3] s=37 -> 0.04 : (s'=29) + 0.84 : (s'=36) + 0.04 : (s'=38) + 0.04 : (s'=45) + 0.039999999999999925 : (s'=37);
	[a4] s=37 -> 0.84 : (s'=29) + 0.04 : (s'=36) + 0.04 : (s'=38) + 0.04 : (s'=45) + 0.039999999999999925 : (s'=37);
	[a0] s=38 -> 1.0 : (s'=38);
	[a1] s=38 -> 0.04 : (s'=30) + 0.04 : (s'=37) + 0.84 : (s'=39) + 0.04 : (s'=46) + 0.040000000000000036 : (s'=38);
	[a2] s=38 -> 0.04 : (s'=30) + 0.04 : (s'=37) + 0.04 : (s'=39) + 0.84 : (s'=46) + 0.040000000000000036 : (s'=38);
	[a3] s=38 -> 0.04 : (s'=30) + 0.84 : (s'=37) + 0.04 : (s'=39) + 0.04 : (s'=46) + 0.039999999999999925 : (s'=38);
	[a4] s=38 -> 0.84 : (s'=30) + 0.04 : (s'=37) + 0.04 : (s'=39) + 0.04 : (s'=46) + 0.039999999999999925 : (s'=38);
	[a0] s=39 -> 1.0 : (s'=39);
	[a1] s=39 -> 0.04 : (s'=31) + 0.88 : (s'=38) + 0.04 : (s'=47) + 0.039999999999999925 : (s'=39);
	[a2] s=39 -> 0.04 : (s'=31) + 0.08 : (s'=38) + 0.84 : (s'=47) + 0.040000000000000036 : (s'=39);
	[a3] s=39 -> 0.04 : (s'=31) + 0.88 : (s'=38) + 0.04 : (s'=47) + 0.039999999999999925 : (s'=39);
	[a4] s=39 -> 0.84 : (s'=31) + 0.08 : (s'=38) + 0.04 : (s'=47) + 0.040000000000000036 : (s'=39);
	[a0] s=40 -> 1.0 : (s'=40);
	[a1] s=40 -> 0.04 : (s'=32) + 0.88 : (s'=41) + 0.04 : (s'=48) + 0.039999999999999925 : (s'=40);
	[a2] s=40 -> 0.04 : (s'=32) + 0.08 : (s'=41) + 0.84 : (s'=48) + 0.040000000000000036 : (s'=40);
	[a3] s=40 -> 0.04 : (s'=32) + 0.88 : (s'=41) + 0.04 : (s'=48) + 0.039999999999999925 : (s'=40);
	[a4] s=40 -> 0.84 : (s'=32) + 0.08 : (s'=41) + 0.04 : (s'=48) + 0.040000000000000036 : (s'=40);
	[a0] s=41 -> 1.0 : (s'=41);
	[a1] s=41 -> 0.04 : (s'=33) + 0.04 : (s'=40) + 0.84 : (s'=42) + 0.04 : (s'=49) + 0.040000000000000036 : (s'=41);
	[a2] s=41 -> 0.04 : (s'=33) + 0.04 : (s'=40) + 0.04 : (s'=42) + 0.84 : (s'=49) + 0.040000000000000036 : (s'=41);
	[a3] s=41 -> 0.04 : (s'=33) + 0.84 : (s'=40) + 0.04 : (s'=42) + 0.04 : (s'=49) + 0.039999999999999925 : (s'=41);
	[a4] s=41 -> 0.84 : (s'=33) + 0.04 : (s'=40) + 0.04 : (s'=42) + 0.04 : (s'=49) + 0.039999999999999925 : (s'=41);
	[a0] s=42 -> 1.0 : (s'=42);
	[a1] s=42 -> 0.04 : (s'=34) + 0.04 : (s'=41) + 0.84 : (s'=43) + 0.04 : (s'=50) + 0.040000000000000036 : (s'=42);
	[a2] s=42 -> 0.04 : (s'=34) + 0.04 : (s'=41) + 0.04 : (s'=43) + 0.84 : (s'=50) + 0.040000000000000036 : (s'=42);
	[a3] s=42 -> 0.04 : (s'=34) + 0.84 : (s'=41) + 0.04 : (s'=43) + 0.04 : (s'=50) + 0.039999999999999925 : (s'=42);
	[a4] s=42 -> 0.84 : (s'=34) + 0.04 : (s'=41) + 0.04 : (s'=43) + 0.04 : (s'=50) + 0.039999999999999925 : (s'=42);
	[a0] s=43 -> 1.0 : (s'=43);
	[a1] s=43 -> 0.04 : (s'=35) + 0.04 : (s'=42) + 0.84 : (s'=44) + 0.04 : (s'=51) + 0.040000000000000036 : (s'=43);
	[a2] s=43 -> 0.04 : (s'=35) + 0.04 : (s'=42) + 0.04 : (s'=44) + 0.84 : (s'=51) + 0.040000000000000036 : (s'=43);
	[a3] s=43 -> 0.04 : (s'=35) + 0.84 : (s'=42) + 0.04 : (s'=44) + 0.04 : (s'=51) + 0.039999999999999925 : (s'=43);
	[a4] s=43 -> 0.84 : (s'=35) + 0.04 : (s'=42) + 0.04 : (s'=44) + 0.04 : (s'=51) + 0.039999999999999925 : (s'=43);
	[a0] s=44 -> 1.0 : (s'=44);
	[a1] s=44 -> 0.04 : (s'=36) + 0.04 : (s'=43) + 0.84 : (s'=45) + 0.04 : (s'=52) + 0.040000000000000036 : (s'=44);
	[a2] s=44 -> 0.04 : (s'=36) + 0.04 : (s'=43) + 0.04 : (s'=45) + 0.84 : (s'=52) + 0.040000000000000036 : (s'=44);
	[a3] s=44 -> 0.04 : (s'=36) + 0.84 : (s'=43) + 0.04 : (s'=45) + 0.04 : (s'=52) + 0.039999999999999925 : (s'=44);
	[a4] s=44 -> 0.84 : (s'=36) + 0.04 : (s'=43) + 0.04 : (s'=45) + 0.04 : (s'=52) + 0.039999999999999925 : (s'=44);
	[a0] s=45 -> 1.0 : (s'=45);
	[a1] s=45 -> 0.04 : (s'=37) + 0.04 : (s'=44) + 0.84 : (s'=46) + 0.04 : (s'=53) + 0.040000000000000036 : (s'=45);
	[a2] s=45 -> 0.04 : (s'=37) + 0.04 : (s'=44) + 0.04 : (s'=46) + 0.84 : (s'=53) + 0.040000000000000036 : (s'=45);
	[a3] s=45 -> 0.04 : (s'=37) + 0.84 : (s'=44) + 0.04 : (s'=46) + 0.04 : (s'=53) + 0.039999999999999925 : (s'=45);
	[a4] s=45 -> 0.84 : (s'=37) + 0.04 : (s'=44) + 0.04 : (s'=46) + 0.04 : (s'=53) + 0.039999999999999925 : (s'=45);
	[a0] s=46 -> 1.0 : (s'=46);
	[a1] s=46 -> 0.04 : (s'=38) + 0.04 : (s'=45) + 0.84 : (s'=47) + 0.04 : (s'=54) + 0.040000000000000036 : (s'=46);
	[a2] s=46 -> 0.04 : (s'=38) + 0.04 : (s'=45) + 0.04 : (s'=47) + 0.84 : (s'=54) + 0.040000000000000036 : (s'=46);
	[a3] s=46 -> 0.04 : (s'=38) + 0.84 : (s'=45) + 0.04 : (s'=47) + 0.04 : (s'=54) + 0.039999999999999925 : (s'=46);
	[a4] s=46 -> 0.84 : (s'=38) + 0.04 : (s'=45) + 0.04 : (s'=47) + 0.04 : (s'=54) + 0.039999999999999925 : (s'=46);
	[a0] s=47 -> 1.0 : (s'=47);
	[a1] s=47 -> 0.04 : (s'=39) + 0.88 : (s'=46) + 0.04 : (s'=55) + 0.039999999999999925 : (s'=47);
	[a2] s=47 -> 0.04 : (s'=39) + 0.08 : (s'=46) + 0.84 : (s'=55) + 0.040000000000000036 : (s'=47);
	[a3] s=47 -> 0.04 : (s'=39) + 0.88 : (s'=46) + 0.04 : (s'=55) + 0.039999999999999925 : (s'=47);
	[a4] s=47 -> 0.84 : (s'=39) + 0.08 : (s'=46) + 0.04 : (s'=55) + 0.040000000000000036 : (s'=47);
	[a0] s=48 -> 1.0 : (s'=48);
	[a1] s=48 -> 0.04 : (s'=40) + 0.88 : (s'=49) + 0.04 : (s'=56) + 0.039999999999999925 : (s'=48);
	[a2] s=48 -> 0.04 : (s'=40) + 0.08 : (s'=49) + 0.84 : (s'=56) + 0.040000000000000036 : (s'=48);
	[a3] s=48 -> 0.04 : (s'=40) + 0.88 : (s'=49) + 0.04 : (s'=56) + 0.039999999999999925 : (s'=48);
	[a4] s=48 -> 0.84 : (s'=40) + 0.08 : (s'=49) + 0.04 : (s'=56) + 0.040000000000000036 : (s'=48);
	[a0] s=49 -> 1.0 : (s'=49);
	[a1] s=49 -> 0.04 : (s'=41) + 0.04 : (s'=48) + 0.84 : (s'=50) + 0.04 : (s'=57) + 0.040000000000000036 : (s'=49);
	[a2] s=49 -> 0.04 : (s'=41) + 0.04 : (s'=48) + 0.04 : (s'=50) + 0.84 : (s'=57) + 0.040000000000000036 : (s'=49);
	[a3] s=49 -> 0.04 : (s'=41) + 0.84 : (s'=48) + 0.04 : (s'=50) + 0.04 : (s'=57) + 0.039999999999999925 : (s'=49);
	[a4] s=49 -> 0.84 : (s'=41) + 0.04 : (s'=48) + 0.04 : (s'=50) + 0.04 : (s'=57) + 0.039999999999999925 : (s'=49);
	[a0] s=50 -> 1.0 : (s'=50);
	[a1] s=50 -> 0.04 : (s'=42) + 0.04 : (s'=49) + 0.84 : (s'=51) + 0.04 : (s'=58) + 0.040000000000000036 : (s'=50);
	[a2] s=50 -> 0.04 : (s'=42) + 0.04 : (s'=49) + 0.04 : (s'=51) + 0.84 : (s'=58) + 0.040000000000000036 : (s'=50);
	[a3] s=50 -> 0.04 : (s'=42) + 0.84 : (s'=49) + 0.04 : (s'=51) + 0.04 : (s'=58) + 0.039999999999999925 : (s'=50);
	[a4] s=50 -> 0.84 : (s'=42) + 0.04 : (s'=49) + 0.04 : (s'=51) + 0.04 : (s'=58) + 0.039999999999999925 : (s'=50);
	[a0] s=51 -> 1.0 : (s'=51);
	[a1] s=51 -> 0.04 : (s'=43) + 0.04 : (s'=50) + 0.84 : (s'=52) + 0.04 : (s'=59) + 0.040000000000000036 : (s'=51);
	[a2] s=51 -> 0.04 : (s'=43) + 0.04 : (s'=50) + 0.04 : (s'=52) + 0.84 : (s'=59) + 0.040000000000000036 : (s'=51);
	[a3] s=51 -> 0.04 : (s'=43) + 0.84 : (s'=50) + 0.04 : (s'=52) + 0.04 : (s'=59) + 0.039999999999999925 : (s'=51);
	[a4] s=51 -> 0.84 : (s'=43) + 0.04 : (s'=50) + 0.04 : (s'=52) + 0.04 : (s'=59) + 0.039999999999999925 : (s'=51);
	[a0] s=52 -> 1.0 : (s'=52);
	[a1] s=52 -> 0.04 : (s'=44) + 0.04 : (s'=51) + 0.84 : (s'=53) + 0.04 : (s'=60) + 0.040000000000000036 : (s'=52);
	[a2] s=52 -> 0.04 : (s'=44) + 0.04 : (s'=51) + 0.04 : (s'=53) + 0.84 : (s'=60) + 0.040000000000000036 : (s'=52);
	[a3] s=52 -> 0.04 : (s'=44) + 0.84 : (s'=51) + 0.04 : (s'=53) + 0.04 : (s'=60) + 0.039999999999999925 : (s'=52);
	[a4] s=52 -> 0.84 : (s'=44) + 0.04 : (s'=51) + 0.04 : (s'=53) + 0.04 : (s'=60) + 0.039999999999999925 : (s'=52);
	[a0] s=53 -> 1.0 : (s'=53);
	[a1] s=53 -> 0.04 : (s'=45) + 0.04 : (s'=52) + 0.84 : (s'=54) + 0.04 : (s'=61) + 0.040000000000000036 : (s'=53);
	[a2] s=53 -> 0.04 : (s'=45) + 0.04 : (s'=52) + 0.04 : (s'=54) + 0.84 : (s'=61) + 0.040000000000000036 : (s'=53);
	[a3] s=53 -> 0.04 : (s'=45) + 0.84 : (s'=52) + 0.04 : (s'=54) + 0.04 : (s'=61) + 0.039999999999999925 : (s'=53);
	[a4] s=53 -> 0.84 : (s'=45) + 0.04 : (s'=52) + 0.04 : (s'=54) + 0.04 : (s'=61) + 0.039999999999999925 : (s'=53);
	[a0] s=54 -> 1.0 : (s'=54);
	[a1] s=54 -> 0.04 : (s'=46) + 0.04 : (s'=53) + 0.84 : (s'=55) + 0.04 : (s'=62) + 0.040000000000000036 : (s'=54);
	[a2] s=54 -> 0.04 : (s'=46) + 0.04 : (s'=53) + 0.04 : (s'=55) + 0.84 : (s'=62) + 0.040000000000000036 : (s'=54);
	[a3] s=54 -> 0.04 : (s'=46) + 0.84 : (s'=53) + 0.04 : (s'=55) + 0.04 : (s'=62) + 0.039999999999999925 : (s'=54);
	[a4] s=54 -> 0.84 : (s'=46) + 0.04 : (s'=53) + 0.04 : (s'=55) + 0.04 : (s'=62) + 0.039999999999999925 : (s'=54);
	[a0] s=55 -> 1.0 : (s'=55);
	[a1] s=55 -> 0.04 : (s'=47) + 0.88 : (s'=54) + 0.04 : (s'=63) + 0.039999999999999925 : (s'=55);
	[a2] s=55 -> 0.04 : (s'=47) + 0.08 : (s'=54) + 0.84 : (s'=63) + 0.040000000000000036 : (s'=55);
	[a3] s=55 -> 0.04 : (s'=47) + 0.88 : (s'=54) + 0.04 : (s'=63) + 0.039999999999999925 : (s'=55);
	[a4] s=55 -> 0.84 : (s'=47) + 0.08 : (s'=54) + 0.04 : (s'=63) + 0.040000000000000036 : (s'=55);
	[a0] s=56 -> 1.0 : (s'=56);
	[a1] s=56 -> 0.08 : (s'=48) + 0.88 : (s'=57) + 0.040000000000000036 : (s'=56);
	[a2] s=56 -> 0.88 : (s'=48) + 0.08 : (s'=57) + 0.040000000000000036 : (s'=56);
	[a3] s=56 -> 0.08 : (s'=48) + 0.88 : (s'=57) + 0.040000000000000036 : (s'=56);
	[a4] s=56 -> 0.88 : (s'=48) + 0.08 : (s'=57) + 0.040000000000000036 : (s'=56);
	[a0] s=57 -> 1.0 : (s'=57);
	[a1] s=57 -> 0.08 : (s'=49) + 0.04 : (s'=56) + 0.84 : (s'=58) + 0.040000000000000036 : (s'=57);
	[a2] s=57 -> 0.88 : (s'=49) + 0.04 : (s'=56) + 0.04 : (s'=58) + 0.039999999999999925 : (s'=57);
	[a3] s=57 -> 0.08 : (s'=49) + 0.84 : (s'=56) + 0.04 : (s'=58) + 0.040000000000000036 : (s'=57);
	[a4] s=57 -> 0.88 : (s'=49) + 0.04 : (s'=56) + 0.04 : (s'=58) + 0.039999999999999925 : (s'=57);
	[a0] s=58 -> 1.0 : (s'=58);
	[a1] s=58 -> 0.08 : (s'=50) + 0.04 : (s'=57) + 0.84 : (s'=59) + 0.040000000000000036 : (s'=58);
	[a2] s=58 -> 0.88 : (s'=50) + 0.04 : (s'=57) + 0.04 : (s'=59) + 0.039999999999999925 : (s'=58);
	[a3] s=58 -> 0.08 : (s'=50) + 0.84 : (s'=57) + 0.04 : (s'=59) + 0.040000000000000036 : (s'=58);
	[a4] s=58 -> 0.88 : (s'=50) + 0.04 : (s'=57) + 0.04 : (s'=59) + 0.039999999999999925 : (s'=58);
	[a0] s=59 -> 1.0 : (s'=59);
	[a1] s=59 -> 0.08 : (s'=51) + 0.04 : (s'=58) + 0.84 : (s'=60) + 0.040000000000000036 : (s'=59);
	[a2] s=59 -> 0.88 : (s'=51) + 0.04 : (s'=58) + 0.04 : (s'=60) + 0.039999999999999925 : (s'=59);
	[a3] s=59 -> 0.08 : (s'=51) + 0.84 : (s'=58) + 0.04 : (s'=60) + 0.040000000000000036 : (s'=59);
	[a4] s=59 -> 0.88 : (s'=51) + 0.04 : (s'=58) + 0.04 : (s'=60) + 0.039999999999999925 : (s'=59);
	[a0] s=60 -> 1.0 : (s'=60);
	[a1] s=60 -> 0.08 : (s'=52) + 0.04 : (s'=59) + 0.84 : (s'=61) + 0.040000000000000036 : (s'=60);
	[a2] s=60 -> 0.88 : (s'=52) + 0.04 : (s'=59) + 0.04 : (s'=61) + 0.039999999999999925 : (s'=60);
	[a3] s=60 -> 0.08 : (s'=52) + 0.84 : (s'=59) + 0.04 : (s'=61) + 0.040000000000000036 : (s'=60);
	[a4] s=60 -> 0.88 : (s'=52) + 0.04 : (s'=59) + 0.04 : (s'=61) + 0.039999999999999925 : (s'=60);
	[a0] s=61 -> 1.0 : (s'=61);
	[a1] s=61 -> 0.08 : (s'=53) + 0.04 : (s'=60) + 0.84 : (s'=62) + 0.040000000000000036 : (s'=61);
	[a2] s=61 -> 0.88 : (s'=53) + 0.04 : (s'=60) + 0.04 : (s'=62) + 0.039999999999999925 : (s'=61);
	[a3] s=61 -> 0.08 : (s'=53) + 0.84 : (s'=60) + 0.04 : (s'=62) + 0.040000000000000036 : (s'=61);
	[a4] s=61 -> 0.88 : (s'=53) + 0.04 : (s'=60) + 0.04 : (s'=62) + 0.039999999999999925 : (s'=61);
	[a0] s=62 -> 1.0 : (s'=62);
	[a1] s=62 -> 0.08 : (s'=54) + 0.04 : (s'=61) + 0.84 : (s'=63) + 0.040000000000000036 : (s'=62);
	[a2] s=62 -> 0.88 : (s'=54) + 0.04 : (s'=61) + 0.04 : (s'=63) + 0.039999999999999925 : (s'=62);
	[a3] s=62 -> 0.08 : (s'=54) + 0.84 : (s'=61) + 0.04 : (s'=63) + 0.040000000000000036 : (s'=62);
	[a4] s=62 -> 0.88 : (s'=54) + 0.04 : (s'=61) + 0.04 : (s'=63) + 0.039999999999999925 : (s'=62);
	[a0] s=63 -> 1.0 : (s'=63);
	[a1] s=63 -> 0.08 : (s'=55) + 0.88 : (s'=62) + 0.040000000000000036 : (s'=63);
	[a2] s=63 -> 0.88 : (s'=55) + 0.08 : (s'=62) + 0.040000000000000036 : (s'=63);
	[a3] s=63 -> 0.08 : (s'=55) + 0.88 : (s'=62) + 0.040000000000000036 : (s'=63);
	[a4] s=63 -> 0.88 : (s'=55) + 0.08 : (s'=62) + 0.040000000000000036 : (s'=63);
	[a0] s=64 -> 1.0 : (s'=0);
	[a1] s=64 -> 1.0 : (s'=0);
	[a2] s=64 -> 1.0 : (s'=0);
	[a3] s=64 -> 1.0 : (s'=0);
	[a4] s=64 -> 1.0 : (s'=0);
	[a0] s=65 -> 1.0 : (s'=65);
	[a1] s=65 -> 1.0 : (s'=65);
	[a2] s=65 -> 1.0 : (s'=65);
	[a3] s=65 -> 1.0 : (s'=65);
	[a4] s=65 -> 1.0 : (s'=65);

endmodule

init s = 64 endinit

