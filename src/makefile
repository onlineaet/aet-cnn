
.PHONY = all install clean

CFLAGS    :=
TARGETMAIN  = aitest
#替换为你的aet编译器的安装目录
GCC_AET_INSTALL_DIR  :=/home/xx/aetgcc/install

GCC_AET_BIN_DIR      :=$(GCC_AET_INSTALL_DIR)/bin

TOPDIR := $(shell pwd) 
TOPDIR := $(dir $(TOPDIR))

INCLUDE_DIR             =/usr/include
AET_INCLUDE_DIR         =$(TOPDIR)/src 
LIB_DIR                 =/usr/lib
OBJECTDIR					=$(TOPDIR)/build
BIN_DIR                 =$(TOPDIR)/bin

AS      = $(GCC_AET_BIN_DIR)/as
LD      = $(GCC_AET_BIN_DIR)/ld
CC      = $(GCC_AET_BIN_DIR)/gcc 
CPP     = $(CC) -E
AR      = $(GCC_AET_BIN_DIR)/ar
NM      = $(GCC_AET_BIN_DIR)/nm
STRIP   =strip
RANLIB  = $(GCC_AET_BIN_DIR)/ranlib

LIBS	  +=-L$(LIB_DIR) 
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

CFLAGS          +=-O3 -ffast-math -march=native -fPIC -MD $(INCLUDEDIR)

#把 $(SOURCES)中的所有的.c换成.o
SRCOBJS   := $(patsubst %.c,%.o,$(SOURCES))
BUILDOBJS := $(subst $(SRCDIR),$(OBJECTDIR),$(SRCOBJS))

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
