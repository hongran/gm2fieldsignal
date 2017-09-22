all:
	make -C Dsp
	make install -C Dsp
	make -C Fid
	make install -C Fid
	make -C Barcode
	make install -C Barcode

%:
	make $@ -C Dsp
	make $@ -C Fid
	make $@ -C Barcode
