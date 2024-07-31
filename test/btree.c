#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_KEYS 3
#define MAX_CHILDREN (MAX_KEYS + 1)

typedef struct BTreeNode {
    int keys[MAX_KEYS];
    struct BTreeNode* children[MAX_CHILDREN];
    int num_keys;
    bool is_leaf;
} BTreeNode;

BTreeNode* create_node(bool is_leaf) {
    BTreeNode* new_node = (BTreeNode*)malloc(sizeof(BTreeNode));
    new_node->is_leaf = is_leaf;
    new_node->num_keys = 0;
    for (int i = 0; i < MAX_CHILDREN; i++) {
        new_node->children[i] = NULL;
    }
    return new_node;
}

void insert_non_full(BTreeNode* node, int key) {
    int i = node->num_keys - 1;

    if (node->is_leaf) {
        while (i >= 0 && key < node->keys[i]) {
            node->keys[i + 1] = node->keys[i];
            i--;
        }
        node->keys[i + 1] = key;
        node->num_keys++;
    } else {
        while (i >= 0 && key < node->keys[i]) {
            i--;
        }
        i++;
        
        if (node->children[i]->num_keys == MAX_KEYS) {
            // Split child if full
            BTreeNode* new_node = create_node(node->children[i]->is_leaf);
            int mid = MAX_KEYS / 2;
            
            for (int j = 0; j < mid; j++) {
                new_node->keys[j] = node->children[i]->keys[j + mid + 1];
                new_node->children[j] = node->children[i]->children[j + mid + 1];
            }
            new_node->num_keys = mid;
            node->children[i]->num_keys = mid;
            
            for (int j = node->num_keys; j > i; j--) {
                node->keys[j] = node->keys[j - 1];
                node->children[j + 1] = node->children[j];
            }
            node->keys[i] = node->children[i]->keys[mid];
            node->children[i + 1] = new_node;
            node->num_keys++;
            
            if (key > node->keys[i]) {
                i++;
            }
        }
        insert_non_full(node->children[i], key);
    }
}

void insert(BTreeNode** root, int key) {
    if ((*root)->num_keys == MAX_KEYS) {
        BTreeNode* new_root = create_node(false);
        new_root->children[0] = *root;
        *root = new_root;
        insert_non_full(*root, key);
    } else {
        insert_non_full(*root, key);
    }
}

void print_btree(BTreeNode* node, int level) {
    if (node == NULL) return;

    for (int i = node->num_keys - 1; i >= 0; i--) {
        print_btree(node->children[i + 1], level + 1);
        for (int j = 0; j < level; j++) printf("    ");
        printf("%d\n", node->keys[i]);
    }
    print_btree(node->children[0], level + 1);
}

int main() {
    BTreeNode* root = create_node(true);
    
    int keys[] = {3, 7, 1, 5, 11, 17, 13, 19, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    int num_keys = sizeof(keys) / sizeof(keys[0]);
    
    for (int i = 0; i < num_keys; i++) {
        insert(&root, keys[i]);
    }
    
    printf("B-tree structure:\n");
    print_btree(root, 0);
    
    return 0;
}