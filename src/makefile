#为gcc 输出aet的debug信息设环境变量
.PHONY = all install clean

WERROR?=FALSE
CFLAGS    :=
TARGETMAIN  = aitest

#GCC_AET_INSTALL_DIR  :=/home/sns/gcc153/gccnvptx
#GCC_AET_INSTALL_DIR  :=/home/sns/workspace/gccaet152/install
GCC_AET_INSTALL_DIR  :=/home/sns/temp/testaet2/install

GCC_AET_BIN_DIR      :=$(GCC_AET_INSTALL_DIR)/bin

TOPDIR := $(shell pwd) 
TOPDIR := $(dir $(TOPDIR))

INCLUDE_DIR             =/usr/include
AET_INCLUDE_DIR         =$(TOPDIR)/src 
LIB_DIR                 =/usr/lib
OBJECTDIR					=$(TOPDIR)/build
BIN_DIR                 =$(TOPDIR)/bin

DEBUG=false

AS      = as
LD      = ld
CC      = gcc
CPP     = $(CC) -E
AR      = ar
NM      = nm
STRIP   = strip
RANLIB     = ranlib


AS      = $(GCC_AET_BIN_DIR)/as
LD      = $(GCC_AET_BIN_DIR)/ld
CC      = $(GCC_AET_BIN_DIR)/gcc 
CPP     = $(CC) -E
AR      = $(GCC_AET_BIN_DIR)/ar
NM      = $(GCC_AET_BIN_DIR)/nm
STRIP   =strip
RANLIB  = $(GCC_AET_BIN_DIR)/ranlib

LIBS	  +=-L$(LIB_DIR) 

INCLUDEDIR :=-I$(INCLUDE_DIR) \
    -I$(INCLUDE_DIR)/libxml2 \
    -I$(INCLUDE_DIR)/freetype2 \

CFLAGS     +=-Wimplicit-function-declaration -Wall  -Wno-unused-result -Wno-unknown-pragmas 

ifeq ($(WERROR),true)
CFLAGS    +=-Werror=implicit
CFLAGS    +=-Wstrict-prototypes
CFLAGS    +=-Werror=missing-prototypes -Werror=implicit-function-declaration 
CFLAGS    +=-Werror=pointer-arith -Werror=init-self -Werror=format=2 -Werror=missing-include-dirs -Werror=pointer-to-int-cast
CFLAGS    +=-Werror=return-type
endif

SRCDIR         = $(TOPDIR)/src

CP        := cp
RM        := rm
MV        := mv
MKDIR    := mkdir
SED      := sed
FIND    := find
MKDIR    := mkdir
XARGS    := xargs



SOURCES  +=$(shell find $(SRCDIR)/com/ai -name '*.c')
##main 函数
SOURCES      +=  $(SRCDIR)//test-train.c
INCLUDEDIR      +=-I$(GCC_AET_INSTALL_DIR)/include/libaet
INCLUDEDIR      +=-I$(SRCDIR)
INCLUDEDIR      +=-I/usr/local/cuda/include
LIBS	          =-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lnvJitLink -lgomp -lm -pthread -ldl
LIBS           +=-L$(GCC_AET_INSTALL_DIR)/lib64 -laet -laet_cuda

#要加 -march=native  否则报错：inlining failed in call to ‘always_inline’ ‘_mm256_set1_ps’: target specific option mismatch
CFLAGS          +=-O3 -ffast-math -march=native -fPIC -MD $(INCLUDEDIR)

#把 $(SOURCES)中的所有的.c换成.o
SRCOBJS   := $(patsubst %.c,%.o,$(SOURCES))
BUILDOBJS := $(subst $(SRCDIR),$(OBJECTDIR),$(SRCOBJS))

DEPS            += $(patsubst %.o,%.d,$(BUILDOBJS))


COMPILE_START_TIME = $(shell date +"%Y-%M-%d %H:%M:%S")

DATE        := $(shell date)
# Makefile进入，获取时间戳
START_TIME := $(shell cat /proc/uptime | awk -F "." '{print $$1}')

all:   $(TARGETMAIN)
	@echo 'CFLAGS=$(CFLAGS)'
	@echo 'SOURCES=$(SOURCES)'
	@echo 'SRCDIR=$(SRCDIR)'
	@echo 'TARGET=$(TARGET)'
	@echo 'BUILDOBJS=$(BUILDOBJS)'
	@echo 'LIBS=$(LIBS)'
	@echo 'STARTTIME=$(DATE)'
	@date +"Now time is +%FT%T%z"
	@current_time=`cat /proc/uptime | awk -F "." '{print $$1}'`;\
	time_interval=`expr $${current_time} - $(START_TIME)`;\
	runtime=`date -u -d @$${time_interval} +%Hh:%Mm:%Ss`;\
	echo "######## runtime: $${runtime} ########"
	
-include $(DEPS)

$(TARGETMAIN) :$(BUILDOBJS)
	@echo '链接.o文件------=$(DATE)'
	@$(RM) -f  $(BIN_DIR)/$(TARGETMAIN)
	@$(CC)  $^ $(CFLAGS) -o $@ $(LIBS) $(USER_OBJS) 
	@$(STRIPCMD)

$(OBJECTDIR)%.o: $(SRCDIR)%.c
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS)  -o  $@ -c $<

intall:

clean:
	@$(FIND) $(OBJECTDIR) -name "*.o" -o -name "*.d" -o -name "*.tmp" | $(XARGS) $(RM) -f
	@$(RM) -f $(TARGETMAIN)
	
	


	
