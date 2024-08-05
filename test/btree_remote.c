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