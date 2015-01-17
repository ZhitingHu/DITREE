
PROJECT := ditree

# Petuum
PETUUM_ROOT = /home/zhitingh/github-release/internal_bac
include defns-ditree.mk

# comment it to speedup
DEBUG = 1

PROJECT_DIR := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)
BUILD_DIR = $(PROJECT_DIR)/build

##############################
# DITree
##############################
DITREE_SRCS = $(shell find src -name "*.cpp")
DITREE_HDRS := $(shell find include -name "*.hpp")

# BUILD_INCLUDE_DIR contains any generated header files we want to include.
BUILD_INCLUDE_DIR := $(BUILD_DIR)/src

##############################
# Protocol buffers
##############################
# PROTO_SRCS are the protocol buffer definitions
PROTO_SRC_DIR := src/proto
PROTO_SRCS := $(wildcard $(PROTO_SRC_DIR)/*.proto)
# PROTO_BUILD_DIR will contain the .cc and obj files generated from
# PROTO_SRCS; PROTO_BUILD_INCLUDE_DIR will contain the .h header files
PROTO_BUILD_DIR := $(BUILD_DIR)/$(PROTO_SRC_DIR)
PROTO_BUILD_INCLUDE_DIR := ${BUILD_INCLUDE_DIR}/proto
# The generated files for protocol buffers
PROTO_GEN_HEADER_SRCS := $(addprefix $(PROTO_BUILD_DIR)/, \
		$(notdir ${PROTO_SRCS:.proto=.pb.h}))
PROTO_GEN_HEADER := $(addprefix $(PROTO_BUILD_INCLUDE_DIR)/, \
		$(notdir ${PROTO_SRCS:.proto=.pb.h}))
DITREE_HDRS += $(PROTO_GEN_HEADER)
PROTO_GEN_CC := $(addprefix $(BUILD_DIR)/, ${PROTO_SRCS:.proto=.pb.cc})

# The objects corresponding to the source files
DITREE_OBJS = $(addprefix $(BUILD_DIR)/, ${DITREE_SRCS:.cpp=.o})
PROTO_OBJS := ${PROTO_GEN_CC:.cc=.o}
OBJS := $(PROTO_OBJS) $(DITREE_OBJS)

BUILD_OBJ_DIR = $(BUILD_DIR)/src
BUILD_LIB_DIR = $(BUILD_DIR)/lib
THIRD_PARTY_DIR = ${PROJECT_DIR}/third_party

DITREE_STATIC_NAME = $(BUILD_LIB_DIR)/lib$(PROJECT).a

##############################
# DITree tools
##############################
TOOL_SRCS := $(shell find tools -name "*.cpp")
TOOL_OBJS := $(addprefix $(BUILD_DIR)/, ${TOOL_SRCS:.cpp=.o})
TOOL_BUILD_DIR := $(BUILD_DIR)/tools
TOOL_BINS := ${TOOL_OBJS:.o=.bin}
TOOL_BIN_LINKS := ${TOOL_BINS:.bin=}

##############################
# Flags
##############################
CXX = ${PETUUM_CXX}
CXXFLAGS = ${PETUUM_CXXFLAGS} \
           -std=c++11
           #-std=c++0x

INCFLAGS = ${PETUUM_INCFLAGS} \
           -I${THIRD_PARTY_DIR}/include \
           -I./include \
           -I./src \
           -I${BUILD_INCLUDE_DIR}

LDFLAGS = $(PETUUM_LDFLAGS_DIRS) \
          -L${THIRD_PARTY_DIR}/lib \
          $(PETUUM_LDFLAGS_LIBS) \
          -lprotobuf \
	  -lboost_system \
          -lboost_thread \
          -lpthread

# Debugging
ifeq ($(DEBUG), 1)
  COMMON_FLAGS += -DDEBUG
else
  COMMON_FLAGS += -DNDEBUG
endif

CXXFLAGS += $(COMMON_FLAGS) $(INCFLAGS)
LDFLAGS += $(COMMON_FLAGS) 

##############################
# Set build directories
##############################

ALL_BUILD_DIRS := $(sort \
		$(BUILD_DIR) $(BUILD_LIB_DIR) $(BUILD_OBJ_DIR) \
                $(TOOL_BUILD_DIR) \
		$(PROTO_BUILD_DIR) $(PROTO_BUILD_INCLUDE_DIR))


##############################
# Define build targets
##############################
.PHONY: all clean tools proto

all: $(DITREE_STATIC_NAME) tools

$(ALL_BUILD_DIRS):
	@ mkdir -p $@

$(DITREE_STATIC_NAME): $(OBJS) | $(BUILD_LIB_DIR)
	ar rcs $@ $(OBJS)
	@ echo

$(BUILD_OBJ_DIR)/%.o: src/%.cpp $(DITREE_HDRS) | $(BUILD_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -Wno-unused-result \
	       -c $< -o $@
	@ echo

$(PROTO_BUILD_DIR)/%.pb.o: $(PROTO_BUILD_DIR)/%.pb.cc $(PROTO_GEN_HEADER) \
		| $(PROTO_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@ echo

proto: $(PROTO_GEN_CC) $(PROTO_GEN_HEADER)

$(PROTO_BUILD_DIR)/%.pb.cc $(PROTO_BUILD_DIR)/%.pb.h : \
		$(PROTO_SRC_DIR)/%.proto | $(PROTO_BUILD_DIR)
	protoc --proto_path=$(PROTO_SRC_DIR) --cpp_out=$(PROTO_BUILD_DIR) $<
	@ echo

tools: $(TOOL_BINS) $(TOOL_BIN_LINKS)

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o $(DITREE_STATIC_NAME) $(PETUUM_PS_LIB)
	$(CXX) $< $(DITREE_STATIC_NAME) $(PETUUM_PS_LIB) $(CXXFLAGS) $(INCFLAGS) \
        $(LDFLAGS) -o $@
	@ echo

$(TOOL_BUILD_DIR)/%.o: tools/%.cpp $(DITREE_HDRS) | $(TOOL_BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Wno-unused-result \
               -c $< -o $@
	@ echo

clean:
	@- $(RM) -rf $(ALL_BUILD_DIRS)

.PHONY: clean
