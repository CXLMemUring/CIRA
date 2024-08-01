int remote(BTreeNode** node, int key) {
    if ((*root)->num_keys == MAX_KEYS) {
        BTreeNode* new_root = create_node(false);
        new_root->children[0] = *root;
        *root = new_root;
        insert_non_full(*root, key);
    } else {
        insert_non_full(*root, key);
    }
}