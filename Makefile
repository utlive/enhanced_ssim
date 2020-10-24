# Listing directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# Final target
EXE := $(BIN_DIR)/ssim

# Source files
CFILES := $(wildcard $(SRC_DIR)/*.c)
CPPFILES := $(wildcard $(SRC_DIR)/*.cpp)

# Object files
OBJ := $(CFILES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o) $(CPPFILES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

CPPFLAGS := -Iinclude -MMD -MP -g
CFLAGS := -Wall -g
LDFLAGS := -Llib
LDLIBS := -lm

.PHONY: all clean

all: $(EXE)

$(EXE): $(OBJ) | $(BIN_DIR)
		$(CXX) $(CPPFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
		$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
		$(CXX) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
		mkdir -p $@

clean:
		@$(RM) -rv $(BIN_DIR) $(OBJ_DIR)

-include $(OBJ:.o=.d)
