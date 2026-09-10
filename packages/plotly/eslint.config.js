// Flat config for the one JS package in the repo. Deliberately small: the
// formatting rules live in prettier, so this only carries the checks that
// catch real mistakes -- an unused binding of the kind that let two builders
// destructure a themed `axis` helper and never call it.
export default [
  {
    files: ["src/**/*.js", "test/**/*.js"],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "module",
    },
    linterOptions: { reportUnusedDisableDirectives: true },
    rules: {
      "no-unused-vars": [
        "error",
        // ignoreRestSiblings keeps the destructure-to-omit idiom the fixture
        // test uses to strip the envelope off a payload.
        {
          args: "after-used",
          argsIgnorePattern: "^_",
          ignoreRestSiblings: true,
        },
      ],
      "no-undef": "off",
      eqeqeq: ["error", "always", { null: "ignore" }],
      "no-var": "error",
      "prefer-const": "error",
    },
  },
];
