# MkDocs Migration Plan

## Executive Summary

Plan to migrate DataBeak documentation from Docusaurus to MkDocs Material for
improved Python ecosystem integration, simpler maintenance, and better alignment
with project tooling.

## Current State Analysis

### Docusaurus Setup (Current)

- **Framework**: Docusaurus 3.8.1 with TypeScript
- **Theme**: Default preset-classic with custom CSS
- **Content**: 6 markdown files organized in docs/docs/
- **Features**: Auto-generated sidebar, GitHub integration, responsive design
- **Deployment**: GitHub Pages with `jonpspri.github.io/databeak/`
- **Dependencies**: React 19, complex Node.js ecosystem (290MB)
- **Custom Components**: Minimal (basic homepage features only)

### Documentation Structure

```
docs/docs/
├── intro.md                    # Project introduction
├── installation.md            # Installation guide (critical user path)
├── architecture.md            # Technical architecture
├── VERSION_MANAGEMENT.md       # Version management docs
├── api/
│   └── overview.md            # API reference
└── tutorials/
    └── quickstart.md          # Quick start tutorial
```

## Migration Rationale

### Why MkDocs Material?

#### ✅ **Python Ecosystem Alignment**

- **Native Python tooling**: Integrates seamlessly with uv, pip, and Python dev
  workflow
- **Simplified dependency management**: Already configured in pyproject.toml
- **No Node.js complexity**: Eliminates 290MB+ of JS dependencies
- **Better CI/CD integration**: Faster builds, simpler GitHub Actions

#### ✅ **Maintenance Benefits**

- **Single language stack**: Pure Python eliminates TypeScript maintenance
- **Simpler configuration**: YAML-based vs complex TypeScript config
- **Faster builds**: Python-based generation vs React compilation
- **Reduced complexity**: No package.json, node_modules, or npm concerns

#### ✅ **Feature Parity**

- **Documentation features**: All current features supported
- **Search**: Better search with lunr.js integration
- **Theming**: Material Design with extensive customization
- **Code highlighting**: Superior Python code highlighting with Pygments
- **API documentation**: mkdocstrings for automatic Python API docs

#### ✅ **Developer Experience**

- **Local development**: Faster dev server startup
- **Hot reload**: Instant markdown changes
- **Python integration**: Direct code import and API generation
- **Familiar tooling**: Aligns with existing Python development practices

## Migration Challenges & Solutions

### 🔧 **Technical Challenges**

#### **1. Configuration Migration**

**Challenge**: Convert TypeScript config to YAML **Solution**: Create
`mkdocs.yml` with equivalent settings and plugins

#### **2. Content Conversion**

**Challenge**: Frontmatter and Docusaurus-specific syntax **Solution**: Minimal
\- content is pure markdown with standard frontmatter

#### **3. Custom Styling**

**Challenge**: Preserve current visual design **Solution**: Use Material theme
with custom CSS overrides

#### **4. Homepage Components**

**Challenge**: React components → static content **Solution**: Convert to
markdown/HTML or use Material theme features

### 📋 **Content Migration**

#### **Low Risk Items (Direct Copy)**

- All markdown files (6 total) - pure markdown compatible
- Images and static assets
- Code examples and snippets

#### **Medium Risk Items (Need Adjustment)**

- Homepage layout (React components → markdown)
- Navigation structure (TypeScript → YAML)
- Social/meta tags configuration

### 🚀 **Deployment Changes**

#### **Current**: Docusaurus GitHub Pages

- Build: `npm run build` → `docs/build/`
- Deploy: Docusaurus deploy command
- URL: `jonpspri.github.io/databeak/`

#### **Target**: MkDocs GitHub Pages

- Build: `mkdocs build` → `site/`
- Deploy: GitHub Actions or `mkdocs gh-deploy`
- URL: Same - `jonpspri.github.io/databeak/`

## Detailed Migration Plan

### Phase 1: Setup & Configuration (2-4 hours)

#### **Step 1.1: Create MkDocs Configuration**

```bash
# Already have dependencies in pyproject.toml
# Create mkdocs.yml configuration file
# Configure Material theme with DataBeak branding
```

#### **Step 1.2: Content Structure Mapping**

```
Current: docs/docs/           → Target: docs/
├── intro.md                  → ├── index.md
├── installation.md           → ├── installation.md
├── architecture.md           → ├── architecture.md
├── VERSION_MANAGEMENT.md     → ├── version-management.md
├── api/overview.md           → ├── api/index.md
└── tutorials/quickstart.md   → └── tutorials/quickstart.md
```

#### **Step 1.3: Navigation Configuration**

Convert auto-generated sidebar to explicit YAML navigation structure.

### Phase 2: Content Migration (1-2 hours)

#### **Step 2.1: Markdown Files**

- Copy all .md files to new structure
- Update relative links and paths
- Adjust frontmatter if needed

#### **Step 2.2: Static Assets**

- Migrate images and favicon
- Update asset references in content

#### **Step 2.3: Homepage Creation**

- Convert React components to Material theme homepage
- Preserve key features and CTA structure

### Phase 3: Customization (2-3 hours)

#### **Step 3.1: Theme Configuration**

- Material Design color scheme matching current branding
- Custom CSS for DataBeak-specific styling
- Logo and favicon integration

#### **Step 3.2: Plugin Configuration**

- Code highlighting (Python-focused)
- Search functionality
- Mermaid diagrams (if needed)
- Social cards generation

#### **Step 3.3: API Documentation**

- Set up mkdocstrings for automatic API documentation
- Link to Python source code from docs

### Phase 4: Deployment & Testing (1-2 hours)

#### **Step 4.1: Local Testing**

- Build and test locally: `mkdocs serve`
- Verify all links and navigation work
- Test responsive design

#### **Step 4.2: GitHub Pages Setup**

- Configure GitHub Actions for MkDocs deployment
- Test deployment to gh-pages branch
- Verify URL structure and redirects

#### **Step 4.3: Validation**

- All documentation links working
- Search functionality operational
- Mobile/responsive design verified
- Performance testing (build speed, site speed)

### Phase 5: Cleanup (30 minutes)

#### **Step 5.1: Remove Docusaurus**

- Delete docs/package.json, node_modules/
- Remove Docusaurus configuration files
- Clean up TypeScript dependencies

#### **Step 5.2: Update References**

- Update README.md documentation links (if needed)
- Update contributing guidelines
- Update CI/CD references

## Migration Timeline

### **Estimated Total Time: 6-11 hours**

| Phase                 | Duration   | Dependencies     |
| --------------------- | ---------- | ---------------- |
| Setup & Configuration | 2-4 hours  | -                |
| Content Migration     | 1-2 hours  | Phase 1 complete |
| Customization         | 2-3 hours  | Phase 2 complete |
| Deployment & Testing  | 1-2 hours  | Phase 3 complete |
| Cleanup               | 30 minutes | Phase 4 complete |

### **Critical Path Items**

1. MkDocs configuration with proper navigation
1. Homepage content conversion
1. GitHub Pages deployment configuration
1. Link validation and testing

## Risk Assessment

### **Low Risk ✅**

- **Content migration**: Pure markdown files
- **Basic configuration**: Standard MkDocs setup
- **Styling**: Material theme provides good defaults

### **Medium Risk ⚠️**

- **Homepage design**: Converting React components to markdown
- **Search functionality**: Ensuring feature parity
- **Build pipeline**: New GitHub Actions workflow

### **High Risk 🚨**

- **URL preservation**: Must maintain existing documentation URLs
- **Deployment downtime**: Minimize disruption to users
- **Feature regression**: Ensure no functionality loss

## Success Criteria

### **Must Have**

- [ ] All existing documentation content accessible
- [ ] Equivalent navigation and user experience
- [ ] Same or better search functionality
- [ ] Preserved URL structure (no broken links)
- [ ] Mobile responsive design
- [ ] Fast build and deployment

### **Nice to Have**

- [ ] Improved Python code highlighting
- [ ] Automatic API documentation generation
- [ ] Faster local development server
- [ ] Simplified maintenance workflow
- [ ] Better integration with Python tooling

## Next Steps

1. **Get approval** for migration approach and timeline
1. **Create mkdocs.yml** configuration file
1. **Set up parallel development** environment for testing
1. **Begin content migration** in phases
1. **Test deployment** to ensure smooth transition

## Rollback Plan

If migration encounters critical issues:

1. **Preserve current Docusaurus setup** until MkDocs is fully functional
1. **Parallel deployment** approach - run both systems until MkDocs proven
1. **Quick rollback**: Keep current GitHub Pages deployment active
1. **Documentation links**: Use feature flags or redirects during transition

______________________________________________________________________

**Recommendation**: Proceed with migration. DataBeak will benefit from
simplified tooling, better Python integration, and reduced maintenance overhead
while maintaining all current documentation functionality.
